import http.server
import textwrap
from itertools import groupby
from operator import itemgetter
from pathlib import Path
from typing import Dict, Union

import numpy as np
from Bio.SeqUtils import seq1
from jinja2 import Environment, FileSystemLoader

from abcfold.plots.pae_plot import create_pae_plots
from abcfold.plots.plddt_plot import plot_plddt
from abcfold.processoutput.alphafold3 import AlphafoldOutput
from abcfold.processoutput.boltz import BoltzOutput
from abcfold.processoutput.chai import ChaiOutput

PORT = 8000


def get_plddt_regions(plddts: Union[np.ndarray, list]) -> dict:
    """
    Get the pLDDT regions for the model
    """
    if not isinstance(plddts, np.ndarray):
        plddts = np.array(plddts)

    regions = {}
    # replace none values with -1
    plddts = np.where(plddts is None, -1, plddts)

    v_low = np.where((0 <= plddts) & (plddts <= 50))[0]
    regions["v_low"] = get_regions_helper(v_low)
    low = np.where((plddts > 50) & (plddts < 70))[0]
    regions["low"] = get_regions_helper(low)
    confident = np.where((plddts >= 70) & (plddts < 90))[0]
    regions["confident"] = get_regions_helper(confident)
    v_confident = np.where(plddts >= 90)[0]
    regions["v_high"] = get_regions_helper(v_confident)

    return regions


def get_regions_helper(indices):
    """
    Get the regions from the indices
    """
    regions = []
    for _, g in groupby(enumerate(indices), lambda x: x[0] - x[1]):
        group = map(itemgetter(1), g)
        group = list(map(int, group))
        regions.append((group[0], group[-1]))
    return regions


def get_model_sequence_data(cif_objs) -> dict:
    """
    Get the sequence for each chain and ligand in the model, used internally
    for plotting

    Args:
        cif_objs : A list of CifFile objs

    Returns:
        dict : Chain ID and sequence data
    """
    sequence_data: dict = {}
    for cif_obj in cif_objs:
        sequence_data_ = {}
        for chain in cif_obj.get_chains():
            if cif_obj.check_ligand(chain):
                if chain.id not in sequence_data_:
                    sequence_data_[chain.id] = ""
                sequence_data_[chain.id] += "".join(
                    [atom.id[0] for residue in chain for atom in residue]
                )
            elif cif_obj.check_other(chain, ["dna"]):
                sequence_data_[chain.id] = "".join(
                    [residue.get_resname()[-1] for residue in chain]
                )
            elif cif_obj.check_other(chain, ["rna"]):
                sequence_data_[chain.id] = "".join(
                    [residue.get_resname() for residue in chain]
                )
            else:

                sequence_data_[chain.id] = "".join(
                    [seq1(residue.get_resname()) for residue in chain]
                )
        sequence_data = {
            chain_id: sorted(
                [sequence_data_[chain_id], sequence_data.get(chain_id, "")],
                reverse=True,
            )[0]
            for chain_id in sequence_data_
        }

    return sequence_data


def get_model_data(model, plot_dict, method, plddt_scores, output_dir):
    """
    Get the model data for the output page

    Args:
        model (CifFile): Model object
        plot_dict (dict): Dictionary of plots
        method (str): Method used to generate the model
        output_dir (Path): Path to the output directory
    """
    regions = get_plddt_regions(plddt_scores)
    model_data = {
        "model_id": model.name,
        "model_source": method,
        "model_path": model.pathway.as_posix(),
        "plddt_regions": regions,
        "avg_plddt": model.average_plddt,
        "h_score": model.h_score,
        "clashes": model.clashes_residues,
        "pae_path": Path(plot_dict[model.pathway.as_posix()])
        .relative_to(output_dir)
        .as_posix(),
    }
    return model_data


class NoCacheHTTPRequestHandler(http.server.SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header(
            "Cache-Control", "no-store, no-cache, must-revalidate, max-age=0"
        )
        self.send_header("Pragma", "no-cache")
        self.send_header("Expires", "0")
        super().end_headers()


def plots(outputs: list, output_dir: Path):
    """
    Generate plots for the output of the different programs

    Args:
        outputs (list): List of output objects

    """
    pathway_plots = create_pae_plots(outputs, output_dir=output_dir)
    plddt_plot_input: Dict[str, list] = get_all_cif_files(outputs)

    plot_plddt(plddt_plot_input, output_name=output_dir.joinpath("plddt_plot.html"))

    pathway_plots["plddt"] = str(output_dir.joinpath("plddt_plot.html").resolve())

    return pathway_plots


def render_template(in_file_path, out_file_path, **kwargs):
    """
    Templates the given file with the keyword arguments.

    Args:
        in_file_path (Path): The path to the template.
        out_file_path (Path): The path to output the templated file.
        **kwargs (dict): Variables to use in templating.
    """
    env = Environment(
        loader=FileSystemLoader(in_file_path.parent), keep_trailing_newline=True
    )
    template = env.get_template(in_file_path.name)
    output = template.render(**kwargs)
    with open(str(out_file_path), "w") as f:
        f.write(output)


def output_open_html_script(file_out: str, port: int = 8000):
    """
    Make a python script to open the output HTML file in the default web browser

    Args:
        file_out (str): Path to the output script
        port (int): Port to run the server on
    """

    script = f"""
    import http.server
    import socketserver
    import webbrowser
    import sys

    PORT = {port}

    class NoCacheHTTPRequestHandler(http.server.SimpleHTTPRequestHandler):
        def end_headers(self):
            self.send_header("Cache-Control",
                            "no-store, no-cache, must-revalidate, max-age=0")
            self.send_header("Pragma", "no-cache")
            self.send_header("Expires", "0")
            super().end_headers()

    try:
        with socketserver.TCPServer(("", PORT),
                                    NoCacheHTTPRequestHandler) as httpd:
            print(
                f"Serving at port {PORT}: http://localhost:{PORT}/index.html"
                )
            print("Press Ctrl+C to stop the server")
            webbrowser.open(f"http://localhost:{PORT}/index.html")
            httpd.serve_forever()
    except KeyboardInterrupt:
        print("Server stopped")
        sys.exit(0)
    """

    script = textwrap.dedent(script)
    with open(file_out, "w") as f:
        f.write(script)


def get_all_cif_files(outputs) -> Dict[str, list]:
    method_cif_objs: Dict[str, list] = {}

    for output in outputs:
        if isinstance(output, AlphafoldOutput):
            for seed in output.seeds:
                if "Alphafold3" not in method_cif_objs:
                    method_cif_objs["Alphafold3"] = []
                method_cif_objs["Alphafold3"].extend(output.cif_files[seed])
        elif isinstance(output, BoltzOutput):

            method_cif_objs["Boltz-1"] = output.cif_files
        elif isinstance(output, ChaiOutput):
            method_cif_objs["Chai-1"] = output.cif_files

    return method_cif_objs
