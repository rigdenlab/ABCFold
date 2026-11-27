import json
import logging
import subprocess
import tempfile
from pathlib import Path
from typing import Union

from abcfold.esmfold2.af3_to_esmfold2 import Esmfold2json
from abcfold.esmfold2.check_install import ensure_esmfold2_env

logger = logging.getLogger("logger")

RUNNER = Path(__file__).parent.joinpath("runner_template.py")
# The code default is 200 while the published example passes 100, so the value
# is always sent explicitly.
NUM_SAMPLING_STEPS = 100
NUM_LOOPS = 20


def run_esmfold2(
    input_json: Union[str, Path],
    output_dir: Union[str, Path],
    config: dict,
    save_input: bool = False,
    test: bool = False,
    number_of_models: int = 5,
) -> bool:
    """
    Run ESMFold2 using the input JSON file

    ESMFold2 has no command line interface, so abcfold runs a small script
    inside the isolated environment that drives the Python API.

    Args:
        input_json (Union[str, Path]): Path to the input JSON file
        output_dir (Union[str, Path]): Path to the output directory
        config (dict): Configuration dictionary
        save_input (bool): If True, save the input JSON file and MSA to the output
        directory
        test (bool): If True, run the test command
        number_of_models (int): Number of models to generate

    Returns:
        Bool: True if the ESMFold2 run was successful, False otherwise

    Raises:
        subprocess.CalledProcessError: If the ESMFold2 command returns an error
    """
    input_json = Path(input_json)
    output_dir = Path(output_dir)

    logger.debug("Checking if ESMFold2 is installed")
    env = ensure_esmfold2_env(config=config)

    if test:
        try:
            env.run(generate_esmfold2_test_command())
        except subprocess.CalledProcessError:
            logger.error("ESMFold2 run failed")
            return False
        logger.info("ESMFold2 run complete")
        return True

    with tempfile.TemporaryDirectory() as temp_dir:
        working_dir = Path(temp_dir)
        if save_input:
            logger.info("Saving msa to the output directory")
            working_dir = output_dir
            working_dir.mkdir(parents=True, exist_ok=True)

        esmfold2_json = Esmfold2json(working_dir)
        esmfold2_json.json_to_json(input_json)

        out_file = working_dir.joinpath(f"{input_json.stem}_esmfold2.json")
        esmfold2_json.write_json(out_file)

        options_file = working_dir.joinpath(f"{input_json.stem}_esmfold2_options.json")
        options_file.write_text(
            json.dumps(
                {
                    "model_repo": config["esmfold2_model"],
                    "device": config.get("esmfold2_device", "cuda"),
                    "num_loops": NUM_LOOPS,
                    "num_sampling_steps": NUM_SAMPLING_STEPS,
                    "num_diffusion_samples": number_of_models,
                }
            )
        )

        esmfold2_out_dir = output_dir / "esmfold2_results"
        esmfold2_out_dir.mkdir(parents=True, exist_ok=True)
        logger.info("Running ESMFold2 using seeds: %s", esmfold2_json.seeds)

        try:
            env.run(
                generate_esmfold2_command(out_file, esmfold2_out_dir, options_file)
            )
        except subprocess.CalledProcessError as e:
            stderr = e.stderr or ""
            if stderr:
                if working_dir.exists():
                    output_err_file = working_dir / "esmfold2_error.log"
                else:
                    output_err_file = working_dir.parent / "esmfold2_error.log"
                output_err_file.write_text(stderr)
                logger.error(
                    "ESMFold2 run failed. Error log is in %s", output_err_file
                )
            else:
                logger.error("ESMFold2 run failed")
            return False

    logger.info("ESMFold2 run complete")
    logger.info("Output files are in %s", output_dir)
    return True


def generate_esmfold2_command(
    input_json: Union[str, Path],
    output_dir: Union[str, Path],
    options_file: Union[str, Path],
) -> list:
    """
    Generate the ESMFold2 command

    Args:
        input_json (Union[str, Path]): Path to the converted input description
        output_dir (Union[str, Path]): Path to the output directory
        options_file (Union[str, Path]): Path to the runtime options file

    Returns:
        list: The ESMFold2 command
    """
    return [
        "python",
        str(RUNNER),
        str(input_json),
        str(output_dir),
        str(options_file),
    ]


def generate_esmfold2_test_command() -> list:
    """
    Generate the test command for ESMFold2

    ESMFold2 has no CLI to interrogate, so the import is the smoke test.

    Args:
        None

    Returns:
        list: The ESMFold2 test command
    """

    return [
        "python",
        "-c",
        "from esm.models.esmfold2 import EsmFold2Model, ESMFold2InputBuilder; "
        "print('esmfold2 ok')",
    ]
