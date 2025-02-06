import configparser
import http.server
import json
import os
import shutil
import socketserver
import sys
import tempfile
import webbrowser
from pathlib import Path
from typing import Dict

from jinja2 import Environment, FileSystemLoader

from abcfold.abc_script_utils import make_dir, setup_logger
from abcfold.add_mmseqs_msa import add_msa_to_json
from abcfold.argparse_utils import (alphafold_argparse_util,
                                    boltz_argparse_util, chai_argparse_util,
                                    custom_template_argpase_util,
                                    main_argpase_util, mmseqs2_argparse_util,
                                    prediction_argparse_util)
from abcfold.plots.pae_plot import create_pae_plots
from abcfold.plots.plddt_plot import plot_plddt
from abcfold.processoutput.alphafold3 import AlphafoldOutput
from abcfold.processoutput.boltz import BoltzOutput
from abcfold.processoutput.chai import ChaiOutput
from abcfold.run_alphafold3 import run_alphafold3

logger = setup_logger()

HTML_DIR = Path(__file__).parent / 'html'
HTML_TEMPLATE = HTML_DIR.joinpath('abcfold.html.jinja2')
PLOTS_DIR = ".plots"
PORT = 8000


def run(args, config, defaults, config_file):
    """Run ABCFold

    Args:
        args (argparse.Namespace): Arguments from the command line
        config (configparser.SafeConfigParser): Config parser object
        defaults (dict): Default values from the config file
        config_file (Path): Path to the config file


    Raises:
        SystemExit: If the database directory or model parameters directory is not found


    """
    outputs = []

    args.output_dir = Path(args.output_dir)

    make_dir(args.output_dir, overwrite=args.override)
    make_dir(args.output_dir.joinpath(PLOTS_DIR))

    updated_config = False
    if args.model_params != defaults["model_params"]:
        config.set("Databases", "model_params", args.model_params)
        updated_config = True
    if args.database_dir != defaults["database_dir"]:
        config.set("Databases", "database_dir", args.database_dir)
        updated_config = True
    if updated_config:
        with open(config_file, "w") as f:
            config.write(f)

    if not args.database_dir or not Path(args.database_dir).exists():
        logger.error(f"Database directory not found: {args.database_dir}")
        sys.exit(1)
    elif not args.model_params or not Path(args.model_params).exists():
        logger.error(f"Model parameters directory not found: {args.model_params}")
        sys.exit(1)

    with open(args.input_json, "r") as f:
        input_params = json.load(f)

    name = input_params.get("name")
    if name is None:
        logger.error("Input JSON must contain a 'name' field")
        sys.exit(1)

    if args.boltz1:
        from abcfold.boltz1.check_install import check_boltz1

        check_boltz1()
    if args.chai1:
        from abcfold.chai1.check_install import check_chai1

        check_chai1()

    with tempfile.TemporaryDirectory() as temp_dir_str:
        temp_dir = Path(temp_dir_str)
        if args.mmseqs2:
            if not args.output_json:
                input_json = Path(args.input_json)
                run_json = temp_dir.joinpath(
                    input_json.name.replace(".json", "_mmseqs.json")
                )
            else:
                run_json = Path(args.output_json)

            input_params = add_msa_to_json(
                input_json=input_json,
                templates=args.templates,
                num_templates=args.num_templates,
                custom_template=args.custom_template,
                custom_template_chain=args.custom_template_chain,
                target_id=args.target_id,
                input_params=input_params,
                output_json=run_json,
                to_file=True,
            )

        else:
            run_json = Path(args.input_json)

        if not args.alphafold3 and not args.boltz1 and not args.chai1:
            logger.info(
                "Neither AlphaFold3, Boltz-1, or Chai-1 selected. Running AlphaFold3 \
by default"
            )
            args.alphafold3 = True

        if args.alphafold3:

            run_alphafold3(
                input_json=run_json,
                output_dir=args.output_dir,
                model_params=args.model_params,
                database_dir=args.database_dir,
                number_of_models=args.number_of_models,
                num_recycles=args.num_recycles,
            )

            # Need to find the name of the af3_dir
            af3_out_dir = list(args.output_dir.iterdir())[0]
            ao = AlphafoldOutput(af3_out_dir, input_params, name)
            outputs.append(ao)

        if args.boltz1:
            from abcfold.run_boltz import run_boltz

            run_boltz(
                input_json=run_json,
                output_dir=args.output_dir,
                save_input=args.save_input,
                number_of_models=args.number_of_models,
                num_recycles=args.num_recycles,
            )
            bolt_out_dir = list(args.output_dir.glob("boltz_results*"))[0]
            bo = BoltzOutput(bolt_out_dir, input_params, name)
            bo.add_plddt_to_cif()
            outputs.append(bo)

        if args.chai1:
            from abcfold.run_chai1 import run_chai

            chai_output_dir = args.output_dir.joinpath("chai1")
            run_chai(
                input_json=run_json,
                output_dir=chai_output_dir,
                save_input=args.save_input,
                number_of_models=args.number_of_models,
                num_recycles=args.num_recycles,
            )

            co = ChaiOutput(chai_output_dir, input_params, name)
            outputs.append(co)

        plot_dict = plots(outputs, args.output_dir.joinpath(PLOTS_DIR))

        # Compile data to make output page

        alphafold_models = {'models': []}
        if args.alphafold3:
            sequence_data = None
            for seed in ao.output.keys():
                for idx in ao.output[seed].keys():
                    model = ao.output[seed][idx]['cif']
                    model.check_clashes()
                    if sequence_data is None:
                        sequence_data = model.get_model_sequence_data()
                    model_data = {"model_id": model.name,
                                  "model_source": "AlphaFold3",
                                  "model_path": model.pathway.as_posix(),
                                  "plddt_regions": model.plddt_regions,
                                  "avg_plddt": model.average_plddt,
                                  "h_score": model.h_score,
                                  "clashes": model.clashes,
                                  "pae_path": Path(
                                      plot_dict[model.pathway.as_posix()]
                                      ).relative_to(args.output_dir).as_posix()}
                    alphafold_models['models'].append(model_data)

        boltz_models = {'models': []}
        if args.boltz1:
            for idx in bo.output.keys():
                model = bo.output[idx]['cif']
                model.check_clashes()
                if sequence_data is None:
                    sequence_data = model.get_model_sequence_data()
                model_data = {"model_id": model.name,
                              "model_source": "Boltz-1",
                              "model_path": model.pathway.as_posix(),
                              "plddt_regions": model.plddt_regions,
                              "avg_plddt": model.average_plddt,
                              "h_score": model.h_score,
                              "clashes": model.clashes,
                              "pae_path": Path(
                                  plot_dict[model.pathway.as_posix()]
                                  ).relative_to(args.output_dir).as_posix()}
                boltz_models['models'].append(model_data)

        chai_models = {'models': []}
        if args.chai1:
            for idx in co.output.keys():
                if idx >= 0:
                    model = co.output[idx]['cif']
                    model.check_clashes()
                    if sequence_data is None:
                        sequence_data = model.get_model_sequence_data()
                    model_data = {"model_id": model.name,
                                  "model_source": "Chai-1",
                                  "model_path": model.pathway.as_posix(),
                                  "plddt_regions": model.plddt_regions,
                                  "avg_plddt": model.average_plddt,
                                  "h_score": model.h_score,
                                  "clashes": model.clashes,
                                  "pae_path": Path(
                                      plot_dict[model.pathway.as_posix()]
                                      ).relative_to(args.output_dir).as_posix()}
                    chai_models['models'].append(model_data)

        combined_models = alphafold_models["models"] + \
            boltz_models["models"] + chai_models["models"]

        sequence = ""
        for key in sequence_data.keys():
            sequence += sequence_data[key]

        chain_data = {}
        ref = 0
        for key in sequence_data.keys():
            chain_data['Chain ' + key] = (ref, len(sequence_data[key]) + ref - 1)
            ref += len(sequence_data[key])

        results_dict = {"sequence": sequence,
                        "models": combined_models,
                        "plotly_path": Path(plot_dict['plddt']).relative_to(
                            args.output_dir.resolve()).as_posix(),
                        "chain_data": chain_data}
        results_json = json.dumps(results_dict)

        if not args.output_dir.joinpath('.feature_viewer').exists():
            shutil.copytree(HTML_DIR, args.output_dir / '.feature_viewer')

        # Create the index page
        HTML_OUT = args.output_dir.joinpath("index.html")
        html_out = Path(HTML_OUT).resolve()
        render_template(HTML_TEMPLATE, html_out,
                        # kwargs appear as variables in the template
                        abcfold_html_dir='.feature_viewer',
                        results_json=results_json,
                        version=0.1)
        logger.info(f"Output page written to {HTML_OUT}")

        # Change to the output directory to run the server
        os.chdir(args.output_dir)

        # Make a script to open the output HTML file in the default web browser
        output_open_html_script("open_output.py")

        # Start the server
        with socketserver.TCPServer(("", PORT),
                                    http.server.SimpleHTTPRequestHandler) as httpd:
            logger.info(f"Serving at port {PORT}")
            # Open the main HTML page in the default web browser
            webbrowser.open(f"http://localhost:{PORT}/index.html")
            # Keep the server running
            httpd.serve_forever()


def plots(outputs: list, output_dir: Path):
    """
    Generate plots for the output of the different programs

    Args:
        outputs (list): List of output objects

    """
    pathway_plots = create_pae_plots(outputs, output_dir=output_dir)
    plddt_plot_input: Dict[str, list] = {}
    for output in outputs:
        if isinstance(output, AlphafoldOutput):
            for seed in output.seeds:
                if "Alphafold3" not in plddt_plot_input:
                    plddt_plot_input["Alphafold3"] = []
                plddt_plot_input["Alphafold3"].extend(output.cif_files[seed])
        elif isinstance(output, BoltzOutput):

            plddt_plot_input["Boltz-1"] = output.cif_files
        elif isinstance(output, ChaiOutput):
            plddt_plot_input["Chai-1"] = output.cif_files

    plot_plddt(plddt_plot_input, output_name=output_dir.joinpath("plddt_plot.html"))

    pathway_plots["plddt"] = str(output_dir.joinpath("plddt_plot.html").resolve())

    return pathway_plots


def render_template(in_file_path, out_file_path, **kwargs):
    """
    Templates the given file with the keyword arguments.

    Parameters
    ----------
    in_file_path : Path
        The path to the template
    out_file_path : Path
        The path to output the templated file
    **kwargs : dict
        Variables to use in templating
    """
    env = Environment(
        loader=FileSystemLoader(in_file_path.parent),
        keep_trailing_newline=True)
    template = env.get_template(in_file_path.name)
    output = template.render(**kwargs)
    with open(str(out_file_path), "w") as f:
        f.write(output)


def output_open_html_script(file_out: str):
    """
    Make a python script to open the output HTML file in the default web browser
    """

    script = """
    import http.server
    import socketserver
    import webbrowser

    PORT = 8000

    with socketserver.TCPServer(("", PORT),
        http.server.SimpleHTTPRequestHandler) as httpd:
        # Open the main HTML page in the default web browser
        webbrowser.open(f"http://localhost:{PORT}/index.html")
        # Keep the server running
        httpd.serve_forever()
    """

    with open(file_out, "w") as f:
        f.write(script)


def main():
    """
    Run AlphaFold3 / Boltz1 / Chai-1
    """
    import argparse

    parser = argparse.ArgumentParser(description="Run AlphaFold3 / Boltz1 / Chai-1")

    defaults = {}
    config_file = Path(__file__).parent.joinpath("data", "config.ini")
    config = configparser.SafeConfigParser()

    if config_file.exists():
        config.read(str(config_file))
        defaults.update(dict(config.items("Databases")))

    parser = main_argpase_util(parser)
    parser = alphafold_argparse_util(parser)
    parser = boltz_argparse_util(parser)
    parser = chai_argparse_util(parser)
    parser = mmseqs2_argparse_util(parser)
    parser = custom_template_argpase_util(parser)
    parser = prediction_argparse_util(parser)

    parser.set_defaults(**defaults)
    args = parser.parse_args()

    run(
        args,
        config,
        defaults,
        config_file,
    )


if __name__ == "__main__":
    main()
