import logging
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Union

from abcfold.chai1.af3_to_chai import ChaiFasta
from abcfold.chai1.check_install import (ensure_chai_env,
                                         resolve_chai_downloads_dir)

logger = logging.getLogger("logger")
os.environ["DISABLE_PANDERA_IMPORT_WARNING"] = "True"


def run_chai(
    input_json: Union[str, Path],
    output_dir: Union[str, Path],
    config: dict,
    save_input: bool = False,
    test: bool = False,
    number_of_models: int = 5,
    num_recycles: int = 10,
    use_templates_server: bool = False,
    template_hits_path: Path | None = None,
    mmseqs_database: Union[str, Path, None] = None,
) -> bool:
    """
    Run Chai-1 using the input JSON file

    Args:
        input_json (Union[str, Path]): Path to the input JSON file
        output_dir (Union[str, Path]): Path to the output directory
        config (dict): Configuration dictionary
        save_input (bool): If True, save the input fasta file and MSA to the output
        directory
        test (bool): If True, run the test command
        number_of_models (int): Number of models to generate
        num_recycles (int): Number of trunk recycles
        use_templates_server (bool): If True, use templates from the server
        template_hits_path (Path): Path to the template hits m8 file

    Returns:
        Bool: True if the Chai-1 run was successful, False otherwise

    Notes:
        Chai-1's weights/ESM embeddings/conformer cache directory is
        controlled via the 'chai_weights' config option (falling back to
        ~/.chai1), set as the CHAI_DOWNLOADS_DIR environment variable before
        Chai-1 runs, since Chai-1 has no CLI flag for this.
    """
    input_json = Path(input_json)
    output_dir = Path(output_dir)

    logger.debug("Checking if Chai-1 is installed")
    env = ensure_chai_env(config=config)
    kalign_available = env.which("kalign")

    # Can't input chai-1's weights as a cmd line option so we set the
    # CHAI_DOWNLOADS_DIR env var to the resolved path before running
    chai_downloads_dir = resolve_chai_downloads_dir(config)
    chai_downloads_dir.mkdir(parents=True, exist_ok=True)
    os.environ["CHAI_DOWNLOADS_DIR"] = str(chai_downloads_dir)
    logger.info("Chai-1 will use weights/downloads from %s", chai_downloads_dir)

    with tempfile.TemporaryDirectory() as temp_dir:
        working_dir = Path(temp_dir)
        if save_input:
            logger.info("Saving input fasta file and msa to the output directory")
            working_dir = output_dir
            working_dir.mkdir(parents=True, exist_ok=True)

        chai_fasta = ChaiFasta(working_dir, config=config)
        chai_fasta.json_to_fasta(input_json)

        out_fasta = chai_fasta.fasta
        msa_dir = chai_fasta.working_dir
        out_constraints = chai_fasta.constraints

        # Pre-populate PDB template CIFs (skip RCSB re-downloads) and inject any
        # custom/inline templates as fabricated m8 hits served from a local
        # store. Only meaningful when we have templates to hand.
        template_cif_store = None
        if not test:
            import json

            from abcfold.chai1.chai_templates import prepare_chai_templates

            with open(input_json) as f:
                input_params = json.load(f)

            combined_m8, template_cif_store = prepare_chai_templates(
                input_params,
                work_dir=working_dir,
                existing_m8=template_hits_path,
                mmseqs_database=mmseqs_database,
            )
            if combined_m8 is not None:
                template_hits_path = combined_m8

        for seed in chai_fasta.seeds:
            chai_output_dir = output_dir / f"chai_output_seed-{seed}"

            logger.info(f"Running Chai-1 using seed {seed}")
            cmd = (
                generate_chai_command(
                    out_fasta,
                    msa_dir,
                    out_constraints,
                    chai_output_dir,
                    number_of_models,
                    num_recycles=num_recycles,
                    seed=seed,
                    use_templates_server=use_templates_server,
                    template_hits_path=template_hits_path,
                    template_cif_store=template_cif_store,
                    kalign_available=kalign_available,
                )
                if not test
                else generate_chai_test_command()
            )

            try:
                env.run(cmd)
            except subprocess.CalledProcessError as e:
                stderr = e.stderr or ""
                if stderr:
                    if chai_output_dir.exists():
                        output_err_file = chai_output_dir / "chai_error.log"
                    else:
                        output_err_file = chai_output_dir.parent / "chai_error.log"
                    output_err_file.write_text(stderr)
                    logger.error(
                        "Chai-1 run failed. Error log is in %s", output_err_file
                    )
                else:
                    logger.error("Chai-1 run failed")
                return False

        logger.info("Chai-1 run complete")
        return True


def generate_chai_command(
    input_fasta: Union[str, Path],
    msa_dir: Union[str, Path],
    input_constraints: Union[str, Path],
    output_dir: Union[str, Path],
    number_of_models: int = 5,
    num_recycles: int = 10,
    seed: int = 42,
    use_templates_server: bool = False,
    template_hits_path: Path | None = None,
    template_cif_store: Path | None = None,
    kalign_available: bool = True,
) -> list:
    """
    Generate the Chai-1 command

    Args:
        input_fasta (Union[str, Path]): Path to the input fasta file
        msa_dir (Union[str, Path]): Path to the MSA directory
        input_constraints (Union[str, Path]): Path to the input constraints file
        output_dir (Union[str, Path]): Path to the output directory
        number_of_models (int): Number of models to generate
        num_recycles (int): Number of trunk recycles
        seed (int): Seed for the random number generator
        use_templates_server (bool): If True, use templates from the server
        template_hits_path (Path): Path to the template hits m8 file
        template_cif_store (Path): Directory of local template CIFs to serve
        kalign_available (bool): Whether kalign is installed in the Chai env
            (templates need it)

    Returns:
        list: The Chai-1 command

    """

    chai_exe = Path(__file__).parent / "chai.py"
    cmd = [
        "python",
        str(chai_exe),
        "fold",
        str(input_fasta)
    ]

    if Path(msa_dir).exists():
        cmd += ["--msa-directory", str(msa_dir)]
    if Path(input_constraints).exists():
        cmd += ["--constraint-path", str(input_constraints)]

    cmd += ["--num-diffn-samples", str(number_of_models)]
    cmd += ["--num-trunk-recycles", str(num_recycles)]
    cmd += ["--seed", str(seed)]

    assert not (
        use_templates_server and template_hits_path
    ), "Cannot specify both templates server and path"

    wants_templates = use_templates_server or template_hits_path or template_cif_store
    if not kalign_available and wants_templates:
        logger.warning(
            "kalign not found in the Chai-1 environment, skipping template "
            "search. kalign is required for templates with Chai-1."
        )
    else:
        if use_templates_server:
            cmd += ["--use-templates-server"]
        if template_hits_path:
            cmd += ["--template-hits-path", str(template_hits_path)]
        if template_cif_store:
            cmd += ["--template-cif-store", str(template_cif_store)]

    cmd += [str(output_dir)]

    return cmd


def generate_chai_test_command() -> list:
    """
    Generate the Chai-1 test command

    Args:
        None

    Returns:
        list: The Chai-1 test command
    """
    return [
        "chai-lab",
        "fold",
        "--help",
    ]
