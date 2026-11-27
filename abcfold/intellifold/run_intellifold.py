import logging
import subprocess
import tempfile
from pathlib import Path
from typing import Union

from abcfold.intellifold.af3_to_intellifold import Intellifoldjson
from abcfold.intellifold.check_install import ensure_intellifold_env

logger = logging.getLogger("logger")


def run_intellifold(
    input_json: Union[str, Path],
    output_dir: Union[str, Path],
    config: dict,
    save_input: bool = False,
    test: bool = False,
    number_of_models: int = 5,
) -> bool:
    """
    Run IntelliFold using the input JSON file

    Args:
        input_json (Union[str, Path]): Path to the input JSON file
        output_dir (Union[str, Path]): Path to the output directory
        config (dict): Configuration dictionary
        save_input (bool): If True, save the input JSON file and MSA to the output
        directory
        test (bool): If True, run the test command
        number_of_models (int): Number of models to generate

    Returns:
        Bool: True if the IntelliFold run was successful, False otherwise

    Raises:
        subprocess.CalledProcessError: If the IntelliFold command returns an error
    """
    input_json = Path(input_json)
    output_dir = Path(output_dir)

    logger.debug("Checking if IntelliFold is installed")
    env = ensure_intellifold_env(config=config)

    intellifold_weight_dir = config.get("intellifold_weights")
    if intellifold_weight_dir is not None and intellifold_weight_dir != "None":
        model_dir = Path(intellifold_weight_dir)
    else:
        model_dir = Path.home().joinpath(".intellifold", "model_v2")
    # The CLI default is ./model_v2, relative to the working directory, so an
    # absolute path is always passed. Weights are fetched on first use.
    model_dir.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as temp_dir:
        working_dir = Path(temp_dir)
        if save_input:
            logger.info("Saving msa to the output directory")
            working_dir = output_dir
            working_dir.mkdir(parents=True, exist_ok=True)

        intellifold_json = Intellifoldjson(working_dir)
        intellifold_json.json_to_json(input_json)

        out_file = working_dir.joinpath(f"{input_json.stem}_intellifold.json")
        intellifold_json.write_json(out_file)

        intellifold_out_dir = output_dir / "intellifold_results"
        logger.info("Running IntelliFold using seeds: %s", intellifold_json.seeds)
        cmd = (
            generate_intellifold_command(
                out_file,
                intellifold_out_dir,
                model_dir,
                number_of_models,
                run_data_pipeline=not intellifold_json.uses_msa,
            )
            if not test
            else generate_intellifold_test_command()
        )

        try:
            env.run(cmd)
        except subprocess.CalledProcessError as e:
            stderr = e.stderr or ""
            if stderr:
                if working_dir.exists():
                    output_err_file = working_dir / "intellifold_error.log"
                else:
                    output_err_file = working_dir.parent / "intellifold_error.log"
                output_err_file.write_text(stderr)
                logger.error(
                    "IntelliFold run failed. Error log is in %s", output_err_file
                )
            else:
                logger.error("IntelliFold run failed")
            return False

    logger.info("IntelliFold run complete")
    logger.info("Output files are in %s", output_dir)
    return True


def generate_intellifold_command(
    input_json: Union[str, Path],
    output_dir: Union[str, Path],
    model_dir: Union[str, Path],
    number_of_models: int = 5,
    run_data_pipeline: bool = False,
) -> list:
    """
    Generate the IntelliFold command

    Seeds are not a command line flag: the vendored AlphaFold 3 engine reads
    them from ``modelSeeds`` in the input JSON and runs once per seed.
    Everything after the bare ``--`` is forwarded verbatim to the engine's
    absl flags.

    Args:
        input_json (Union[str, Path]): Path to the input JSON file
        output_dir (Union[str, Path]): Path to the output directory
        model_dir (Union[str, Path]): Path to the model weights directory
        number_of_models (int): Number of diffusion samples to generate
        run_data_pipeline (bool): If True, let the engine search for alignments

    Returns:
        list: The IntelliFold command
    """
    cmd = [
        "intellifold",
        "predict",
        str(input_json),
        f"--model-dir={model_dir}",
        f"--output-dir={output_dir}",
        "--",
        f"--num_diffusion_samples={number_of_models}",
        # Without this the engine appends a timestamp when the output
        # directory already exists, which breaks output collection.
        "--force_output_dir",
    ]

    if not run_data_pipeline:
        cmd.extend(["--norun_data_pipeline", "--resolve_msa_overlaps=false"])

    return cmd


def generate_intellifold_test_command() -> list:
    """
    Generate the test command for IntelliFold

    Args:
        None

    Returns:
        list: The IntelliFold test command
    """

    return [
        "intellifold",
        "predict",
        "--help",
    ]
