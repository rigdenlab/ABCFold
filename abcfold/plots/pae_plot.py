# Run the code for the PAE plots

import logging
import shutil
import subprocess
from multiprocessing import Process
from pathlib import Path
from typing import Dict, List, Union

import pandas as pd

from abcfold.processoutput.alphafold3 import AlphafoldOutput
from abcfold.processoutput.boltz import BoltzOutput
from abcfold.processoutput.chai import ChaiOutput
from abcfold.processoutput.file_handlers import CifFile

logger = logging.getLogger(__name__)

CSSPATHS = {
    "A": "paeViewerStandaloneLayoutAF3.css",
    "B": "paeViewerStandaloneLayoutBoltz.css",
    "C": "paeViewerStandaloneLayoutChai.css",
}

PAEVIEWER = Path(__file__).parent.joinpath(
    "pae-viewer-main", "standalone", "pae_viewer.py"
)

CREATETEMPLATE = Path(__file__).parent.joinpath(
    "pae-viewer-main", "standalone", "create_template.py"
)

INCLUDE_EXTS = [
    ".css",
    ".js",
    ".html",
    ".tpl",
    ".svg",
    ".ico",
    ".png",
]


COLUMNS = [
    "Protein1",
    "SeqPos1",
    "Protein2",
    "SeqPos2",
    "RestraintSatisfied",
]


def create_pae_plots(
    outputs: list,
    output_dir: Union[str, Path],
) -> Dict[str, str]:
    """
    Create PAE html plots for the outputs

    Args:
        outputs: List of outputs to create plots for
        output_dir: Output directory for the plots

    Returns:
        Pathway plot dictionary with the key as the plot path and value as the plot path

    Raises:
        ValueError: If the output type is invalid


    """
    pathway_plot: Dict[str, str] = {}
    run_scripts: List[list] = []
    template_files: List[Path] = []
    output_dir = Path(output_dir)

    # start by copying the pae viewer files to the output directory
    copy_pae_viewer_files(output_dir.joinpath(".pae_viewer"))

    for output in outputs:
        plots_dir = (
            Path(output_dir)
            if output_dir
            else output.output_dir.parent.joinpath(".plots")
        )
        plots_dir.mkdir(exist_ok=True)

        if isinstance(output, BoltzOutput):
            css_path = CSSPATHS["B"]
            template_file = plots_dir.joinpath("boltz_template.html")
            template_files.append(template_file)
            cmd = get_template_run_script(
                "ABCFold - Boltz-1 Output",
                css_path,
                template_file,
                output_dir.joinpath(".pae_viewer"),
            )
            run_script(cmd)

        elif isinstance(output, ChaiOutput):
            css_path = CSSPATHS["C"]
            template_file = plots_dir.joinpath("chai_template.html")
            template_files.append(template_file)
            cmd = get_template_run_script(
                "ABCFold - Chai-1 Output",
                css_path,
                template_file,
                output_dir.joinpath(".pae_viewer"),
            )
            run_script(cmd)

        elif isinstance(output, AlphafoldOutput):
            css_path = CSSPATHS["A"]
            template_file = plots_dir.joinpath("af3_template.html")
            template_files.append(template_file)
            cmd = get_template_run_script(
                "ABCFold - AlphaFold-3 Output",
                css_path,
                template_file,
                output_dir.joinpath(".pae_viewer"),
            )
            run_script(cmd)

            for seed in output.seeds:
                run_scripts.extend(
                    prepare_scripts(
                        output.cif_files[seed],
                        output.af3_pae_files[seed],
                        plots_dir,
                        pathway_plot,
                        template_file,
                        True,
                    )
                )

            continue
        else:
            logger.error("Invalid output type")
            raise ValueError()

        run_scripts.extend(
            prepare_scripts(
                output.cif_files,
                output.af3_pae_files,
                plots_dir,
                pathway_plot,
                template_file,
                False,
            )
        )

    processes = [Process(target=run_script, args=(script,)) for script in run_scripts]
    for process in processes:
        process.start()
    for process in processes:
        process.join()

    for template_file in template_files:
        template_file.unlink()

    for csv_file in output_dir.glob("*.csv"):
        csv_file.unlink()

    return pathway_plot


def prepare_scripts(
    cif_files, pae_files, plots_dir, pathway_plot, template_file, is_af3=False
):

    scripts = []
    for cif_file, pae_file in zip(cif_files, pae_files):
        name_stem = f"{pae_file.pathway.stem}\
{'_' + pae_file.pathway.parent.stem if is_af3 else ''}_{'af3_' if is_af3 else ''}"

        clashes_csv_file = plots_dir.joinpath(f"{name_stem}clashes.csv")
        clashes_csv(cif_file, clashes_csv_file)

        labels = [f"Chain-{chain}" for chain in cif_file.chain_lengths()]
        plot_pathway = plots_dir.joinpath(f"{name_stem}pae_plot.html")
        pae_viewer_script = get_pae_run_script(
            cif_file.pathway,
            labels,
            pae_file.pathway,
            plot_pathway,
            template_file,
            clashes_csv_file,
        )
        pathway_plot[str(cif_file.pathway)] = str(plot_pathway)
        scripts.append(pae_viewer_script)
    return scripts


def run_script(cmd):
    with subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE) as proc:
        stdout, stderr = proc.communicate()
        proc.wait()

        if proc.returncode != 0:
            raise subprocess.CalledProcessError(proc.returncode, cmd, stderr)


def get_pae_run_script(
    cif_path: Union[str, Path],
    labels: List[str],
    pae_path: Union[str, Path],
    output_file: Union[str, Path],
    template_file: Union[str, Path],
    clashes_csv_file: Union[str, Path],
):
    """
    Get the command to run the PAE viewer script

    Args:
        cif_path: Path to the CIF file
        labels: List of labels for the chains
        pae_path: Path to the PAE file
        output_file: Path to the output file
        template_file: Path to the template file
        clashes_csv: Path to the clashes CSV file
    Returns:
        Command to run the PAE viewer script
    """

    cif_path = Path(cif_path)
    pae_path = Path(pae_path)
    output_file = Path(output_file)
    template_file = Path(template_file)
    clashes_csv_file = Path(clashes_csv_file)

    cmd = []
    labels_string = ";".join(labels)
    # labels_string = rf'"{labels_string}"'

    cmd.append("python")
    cmd.append(str(PAEVIEWER.resolve()))
    cmd.append("--structure")
    cmd.append(str(cif_path.resolve()))
    cmd.append("--labels")

    cmd.append(labels_string)
    cmd.append("--scores")
    cmd.append(str(pae_path.resolve()))
    cmd.append("--output_file")
    cmd.append(str(output_file.resolve()))
    cmd.append("--template_file")
    cmd.append(str(template_file.resolve()))
    cmd.append("--crosslinks")
    cmd.append(str(clashes_csv_file.resolve()))
    return cmd


def get_template_run_script(
    title: str,
    standalonecss: str,
    output_file: Union[str, Path],
    src_path: Union[str, Path],
):
    """ """
    output_file = Path(output_file)
    src_path = Path(src_path)
    cmd = []
    cmd.append("python")
    cmd.append(str(CREATETEMPLATE.resolve()))
    cmd.append("--title")
    # title = rf'"{title}"'
    cmd.append(title)
    cmd.append("--standalonecss")
    cmd.append(standalonecss)
    cmd.append("--output_file")
    cmd.append(str(output_file.resolve()))
    cmd.append("--src_path")
    cmd.append(str(src_path.resolve()))

    return cmd


def make_dir(output_dir: Union[str, Path]):
    output_dir = Path(output_dir)
    if output_dir.exists():
        return output_dir
    output_dir.mkdir(parents=True)
    return output_dir


def copy_pae_viewer_files(output_dir: Union[str, Path]):
    output_dir = Path(output_dir)
    if not output_dir.exists():
        output_dir.mkdir(parents=True)
    pae_viewer_main_pathway = Path(__file__).parent.joinpath("pae-viewer-main")
    pae_viewer_files = pae_viewer_main_pathway.rglob("*")

    for file_ in pae_viewer_files:
        if file_.suffix in INCLUDE_EXTS:
            # print out the filepathway from the current directory
            subdirs = list(file_.relative_to(pae_viewer_main_pathway).parts)
            create_subdirs(output_dir, subdirs)
            # copy the file to the output directory
            new_file = output_dir.joinpath(*subdirs)
            if not new_file.exists():
                shutil.copy2(file_, new_file)


def create_subdirs(output_dir: Union[str, Path], subdirs: List[str]):
    output_dir = Path(output_dir)
    subdirs = subdirs[:-1]
    new_subdirs = output_dir
    for subdir in subdirs:
        new_subdirs = new_subdirs.joinpath(subdir)
        if not new_subdirs.exists():
            new_subdirs.mkdir()


def clashes_csv(cif_file: CifFile, output_name: Union[str, Path]):
    output_name = Path(output_name)

    clashes = cif_file.check_clashes()
    df = pd.DataFrame(columns=COLUMNS)

    for clash in clashes:
        atom1, atom2 = clash
        chain_id1, chain_id2 = atom1.get_full_id()[2], atom2.get_full_id()[2]
        res1, res2 = atom1.get_full_id()[3][1], atom2.get_full_id()[3][1]
        df.loc[len(df)] = [
            f"Chain-{chain_id1}",
            res1,
            f"Chain-{chain_id2}",
            res2,
            False,
        ]

    df.to_csv(output_name, index=False)
