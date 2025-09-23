import configparser
import json
import os
import re
import shutil
import socketserver
import sys
import tempfile
import webbrowser
from pathlib import Path

from abcfold.alphafold3.run_alphafold3 import run_alphafold3
from abcfold.argparse_utils import (alphafold_argparse_util,
                                    boltz_argparse_util, chai_argparse_util,
                                    custom_template_argpase_util,
                                    main_argpase_util, mmseqs2_argparse_util,
                                    prediction_argparse_util,
                                    raise_argument_errors,
                                    visuals_argparse_util)
from abcfold.html.html_utils import (PORT, NoCacheHTTPRequestHandler,
                                     get_all_cif_files, get_model_data,
                                     get_model_sequence_data,
                                     output_open_html_script, plots,
                                     render_template)
from abcfold.output.alphafold3 import AlphafoldOutput
from abcfold.output.boltz import BoltzOutput
from abcfold.output.chai import ChaiOutput
from abcfold.output.file_handlers import superpose_models
from abcfold.output.utils import (get_gap_indicies, insert_none_by_minus_one,
                                  make_dummy_m8_file)
from abcfold.scripts.abc_script_utils import (check_input_json, make_dir,
                                              make_dummy_af3_db, setup_logger)
from abcfold.scripts.add_mmseqs_msa import add_msa_to_json

logger = setup_logger()

HTML_DIR = Path(__file__).parent / "html"
HTML_TEMPLATE = HTML_DIR.joinpath("abcfold.html.jinja2")
PLOTS_DIR = ".plots"


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

    if args.mmseqs2:
        logger.info("MMSeqs2 Selected, all other MSAs will be ignored")

    make_dir(args.output_dir, overwrite=args.override)
    make_dir(args.output_dir.joinpath(PLOTS_DIR))

    updated_config = False
    if args.model_params is not None and args.model_params != defaults["model_params"]:
        config.set("Databases", "model_params", args.model_params)
        updated_config = True
    if args.database_dir is not None and args.database_dir != defaults["database_dir"]:
        config.set("Databases", "database_dir", args.database_dir)
        updated_config = True
    if updated_config:
        with open(config_file, "w") as f:
            config.write(f)

    args = raise_argument_errors(args)
    # Ensure that the input json file is valid
    args.input_json = check_input_json(
        args.input_json,
        output_dir=args.output_dir,
        use_af3_templates=args.use_af3_template_search,
    )

    with open(args.input_json, "r") as f:
        input_params = json.load(f)

    name = input_params.get("name")
    if name is None:
        logger.error("Input JSON must contain a 'name' field")
        sys.exit(1)

    if args.alphafold3:
        from abcfold.alphafold3.check_install import check_af3_install

        check_af3_install(interactive=False, sif_path=args.sif_path)

    if args.boltz:
        from abcfold.boltz.check_install import check_boltz

        check_boltz()

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
                mmseqs_db=args.mmseqs_database,
                templates=args.templates,
                num_templates=args.num_templates,
                chai_template_output=temp_dir.joinpath("all_chains.m8"),
                custom_template=args.custom_template,
                custom_template_chain=args.custom_template_chain,
                target_id=args.target_id,
                input_params=input_params,
                output_json=run_json,
                to_file=True,
            )

        else:
            run_json = Path(args.input_json)

        successful_runs = []
        if args.alphafold3:
            af3_database = args.database_dir
            if args.mmseqs2 or (
                any(
                    seq.get("protein", {}).get("unpairedMsa")
                    or seq.get("protein", {}).get("unpairedMsaPath")
                    for seq in input_params["sequences"]
                )
            ):
                af3_database = make_dummy_af3_db(temp_dir)

            af3_success = run_alphafold3(
                input_json=run_json,
                output_dir=args.output_dir,
                model_params=args.model_params,
                database_dir=af3_database,
                number_of_models=args.number_of_models,
                num_recycles=args.num_recycles,
                sif_path=args.sif_path,
            )

            if af3_success:
                pattern = re.compile(f"^{name}.*", re.IGNORECASE)
                af3_out_dir = list(
                    [
                        dir_
                        for dir_ in args.output_dir.glob("*")
                        if dir_.is_dir() and pattern.match(dir_.name)
                    ]
                )[0]
                ao = AlphafoldOutput(af3_out_dir, input_params, name)
                outputs.append(ao)
                run_json = ao.input_json
            successful_runs.append(af3_success)

        if args.boltz:
            from abcfold.boltz.run_boltz import run_boltz

            boltz_success = run_boltz(
                input_json=run_json,
                output_dir=args.output_dir,
                save_input=args.save_input,
                number_of_models=args.number_of_models,
                num_recycles=args.num_recycles,
            )

            if boltz_success:
                bolt_out_dirs = list(args.output_dir.glob("boltz_results*"))
                bo = BoltzOutput(bolt_out_dirs, input_params, name, args.save_input)
                outputs.append(bo)
            successful_runs.append(boltz_success)

        if args.chai1:
            from abcfold.chai1.run_chai1 import run_chai

            template_hits_path = None
            if args.templates and args.mmseqs2:
                template_hits_path = temp_dir.joinpath("all_chain.m8")
            elif args.templates:
                template_hits_path = make_dummy_m8_file(run_json, temp_dir)

            chai_success = run_chai(
                input_json=run_json,
                output_dir=args.output_dir,
                save_input=args.save_input,
                number_of_models=args.number_of_models,
                num_recycles=args.num_recycles,
                template_hits_path=template_hits_path,
            )

            if chai_success:
                chai_output_dirs = list(args.output_dir.glob("chai_output*"))
                co = ChaiOutput(chai_output_dirs, input_params, name, args.save_input)
                outputs.append(co)
            successful_runs.append(chai_success)

        if args.no_visuals:
            logger.info("Visuals disabled")
            return

        if not any(successful_runs):
            logger.error("No models were generated")
            return

        plot_dict = plots(outputs, args.output_dir.joinpath(PLOTS_DIR))

        # Compile data to make output page
        programs_run = []
        cif_models = [
            cif_file
            for cif_list in get_all_cif_files(outputs).values()
            for cif_file in cif_list
        ]
        indicies = get_gap_indicies(*cif_models)
        index_counter = 0

        alphafold_models = {"models": []}

        if args.alphafold3:
            if af3_success:
                programs_run.append("AlphaFold3")
                for seed in ao.output.keys():
                    for idx in ao.output[seed].keys():
                        model = ao.output[seed][idx]["cif"]
                        model.check_clashes()
                        score_file = ao.output[seed][idx]["summary"]
                        plddt = model.residue_plddts
                        if len(indicies) > 0:
                            plddt = insert_none_by_minus_one(
                                indicies[index_counter], plddt
                            )
                        index_counter += 1
                        model_data = get_model_data(
                            model,
                            plot_dict,
                            "AlphaFold3",
                            plddt,
                            score_file,
                            args.output_dir,
                        )
                        alphafold_models["models"].append(model_data)

        boltz_models = {"models": []}
        if args.boltz:
            if boltz_success:
                programs_run.append("Boltz")
                for seed in bo.output.keys():
                    for idx in bo.output[seed].keys():
                        model = bo.output[seed][idx]["cif"]
                        model.check_clashes()
                        score_file = bo.output[seed][idx]["json"]
                        plddt = model.residue_plddts
                        if len(indicies) > 0:
                            plddt = insert_none_by_minus_one(
                                indicies[index_counter], plddt
                            )
                        index_counter += 1
                        model_data = get_model_data(
                            model,
                            plot_dict,
                            "Boltz",
                            plddt,
                            score_file,
                            args.output_dir
                        )
                        boltz_models["models"].append(model_data)

        chai_models = {"models": []}
        if args.chai1:
            if chai_success:
                programs_run.append("Chai-1")
                for seed in co.output.keys():
                    for idx in co.output[seed].keys():
                        if idx >= 0:
                            model = co.output[seed][idx]["cif"]
                            model.check_clashes()
                            score_file = co.output[seed][idx]["scores"]
                            plddt = model.residue_plddts
                            if len(indicies) > 0:
                                plddt = insert_none_by_minus_one(
                                    indicies[index_counter], plddt
                                )
                            index_counter += 1
                            model_data = get_model_data(
                                model,
                                plot_dict,
                                "Chai-1",
                                plddt,
                                score_file,
                                args.output_dir,
                            )
                            chai_models["models"].append(model_data)

        combined_models = (
            alphafold_models["models"] + boltz_models["models"] + chai_models["models"]
        )

        # Make the output directory for the models
        os.makedirs(args.output_dir.joinpath("output_models"), exist_ok=True)
        output_models = []
        for model in combined_models:
            cif_file = args.output_dir.joinpath(model["model_path"])
            if model["model_source"] == "AlphaFold3":
                output_name = "af3_model_" + model["model_id"][-1] + ".cif"
            elif model["model_source"] == "Boltz":
                output_name = "boltz_model_" + model["model_id"][-1] + ".cif"
            elif model["model_source"] == "Chai-1":
                output_name = "chai_model_" + model["model_id"][-1] + ".cif"
            shutil.copy(
                cif_file,
                args.output_dir.joinpath("output_models").joinpath(output_name),
            )
            output_models.append(
                args.output_dir.joinpath("output_models").joinpath(output_name)
            )
        # Superpose the models
        if len(output_models) > 1:
            superpose_models(output_models)

        sequence_data = get_model_sequence_data(cif_models)
        sequence = ""
        for key in sequence_data.keys():
            sequence += sequence_data[key]
        chain_data = {}
        ref = 0
        for key in sequence_data.keys():
            chain_data["Chain " + key] = (ref, len(sequence_data[key]) + ref - 1)
            ref += len(sequence_data[key])
        results_dict = {
            "sequence": sequence,
            "models": combined_models,
            "plotly_path": Path(plot_dict["plddt"])
            .relative_to(args.output_dir.resolve())
            .as_posix(),
            "chain_data": chain_data,
        }
        results_json = json.dumps(results_dict)

        if not args.output_dir.joinpath(".feature_viewer").exists():
            shutil.copytree(HTML_DIR, args.output_dir / ".feature_viewer")

        if len(programs_run) > 1:
            programs = (
                "Structure predictions for: "
                + ", ".join(programs_run[:-1])
                + " and "
                + programs_run[-1]
            )
        else:
            programs = "Structure predictions for: " + programs_run[0]

        # Create the index page
        HTML_OUT = args.output_dir.joinpath("index.html")
        html_out = Path(HTML_OUT).resolve()
        render_template(
            HTML_TEMPLATE,
            html_out,
            # kwargs appear as variables in the template
            abcfold_html_dir=".feature_viewer",
            programs=programs,
            results_json=results_json,
            version=0.1,
        )
        logger.info(f"Output page written to {HTML_OUT}")

        # Change to the output directory to run the server
        os.chdir(args.output_dir)

        # Make a script to open the output HTML file in the default web browser
        output_open_html_script("open_output.py", port=PORT)

        if args.no_server:
            logger.info("Server disabled")
            logger.info(
                "Run 'python open_output.py' in the output directory to \
view the output pages"
            )
            return

        try:
            # Start the server
            with socketserver.TCPServer(("", PORT), NoCacheHTTPRequestHandler) as httpd:
                logger.info(
                    f"Serving at port {PORT}: http://localhost:{PORT}/index.html"
                )
                logger.info("Press Ctrl+C to stop the server")
                # Open the main HTML page in the default web browser
                webbrowser.open(f"http://localhost:{PORT}/index.html")
                # Keep the server running
                httpd.serve_forever()
        except KeyboardInterrupt:
            logger.info("Server stopped")
            httpd.server_close()
            sys.exit(0)


def main():
    """
    Run AlphaFold3 / Boltz / Chai-1
    """
    import argparse

    parser = argparse.ArgumentParser(description="Run AlphaFold3 / Boltz / Chai-1")

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
    parser = visuals_argparse_util(parser)

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
