import logging
import subprocess
import tempfile
from pathlib import Path
from typing import Union

from abcfold.rosettafold3.af3_to_rosettafold3 import Rosettafoldjson
from abcfold.rosettafold3.check_install import (CHECKPOINT_NAME,
                                                ensure_rosettafold_checkpoint,
                                                ensure_rosettafold_env)

logger = logging.getLogger("logger")


def run_rosettafold(
    input_json: Union[str, Path],
    output_dir: Union[str, Path],
    config: dict,
    save_input: bool = False,
    test: bool = False,
    number_of_models: int = 5,
) -> bool:
    """
    Run RosettaFold3 using the input JSON file

    Args:
        input_json (Union[str, Path]): Path to the input JSON file
        output_dir (Union[str, Path]): Path to the output directory
        config (dict): Configuration dictionary
        save_input (bool): If True, save the input JSON file and MSA to the output
        directory
        test (bool): If True, run the test command
        number_of_models (int): Number of models to generate

    Returns:
        Bool: True if the RosettaFold3 run was successful, False otherwise

    Raises:
        subprocess.CalledProcessError: If the RosettaFold3 command returns an error

    """
    input_json = Path(input_json)
    output_dir = Path(output_dir)

    logger.debug("Checking if RosettaFold3 is installed")
    env = ensure_rosettafold_env(config=config)

    rosettafold_weight_dir = config['rosettafold_weights']
    if rosettafold_weight_dir is not None and rosettafold_weight_dir != "None":
        cache_path = Path(rosettafold_weight_dir)
    else:
        cache_path = Path.home().joinpath(".rosettafold3")

    default_ckpt = cache_path.joinpath(CHECKPOINT_NAME)
    if not default_ckpt.exists():
        logger.info(
            "No Checkpoint file found. "
            f"Downloading RosettaFold3 checkpoint to {default_ckpt}"
        )
        rosettafold_ckpt = ensure_rosettafold_checkpoint(default_ckpt)
    else:
        rosettafold_ckpt = default_ckpt

    with tempfile.TemporaryDirectory() as temp_dir:
        working_dir = Path(temp_dir)
        if save_input:
            logger.info("Saving msa to the output directory")
            working_dir = output_dir

        rosettafold_json = Rosettafoldjson(working_dir)
        rosettafold_json.json_to_json(input_json)

        for seed in rosettafold_json.seeds:
            out_file = working_dir.joinpath(f"{input_json.stem}_seed-{seed}.json")

            rosettafold_json.write_json(out_file)
            logger.info("Running RosettaFold3 using seed: %s", seed)
            rosettafold_out_dir = output_dir / f"rosettafold_results_seed-{seed}"
            cmd = (
                generate_rosettafold_command(
                    out_file,
                    rosettafold_out_dir,
                    rosettafold_ckpt,
                    number_of_models,
                    seed
                )
                if not test
                else generate_rosettafold_test_command()
            )

            try:
                env.run(cmd)
            except subprocess.CalledProcessError as e:
                stderr = e.stderr or ""
                if stderr:
                    if working_dir.exists():
                        output_err_file = working_dir / "rosettafold_error.log"
                    else:
                        output_err_file = working_dir.parent / "rosettafold_error.log"
                    output_err_file.write_text(stderr)
                    logger.error(
                        "RosettaFold3 run failed. Error log is in %s", output_err_file
                    )
                else:
                    logger.error("RosettaFold3 run failed")
                return False

    logger.info("RosettaFold3 run complete")
    logger.info("Output files are in %s", output_dir)
    return True


def generate_rosettafold_command(
    input_json: Union[str, Path],
    output_dir: Union[str, Path],
    ckpt_path: Union[str, Path],
    number_of_models: int = 5,
    seed: int = 42,
) -> list:
    """
    Generate the RosettaFold3 command

    Args:
        input_json (Union[str, Path]): Path to the input JSON file
        output_dir (Union[str, Path]): Path to the output directory
        ckpt_path (Union[str, Path]): Path to the inference CheckPoint
        number_of_models (int): Number of models to generate
        seed (int): Random seed to use for the RosettaFold3 run

    Returns:
        list: The RosettaFold3 command
    """
    return [
        "rf3",
        "fold",
        f"inputs='{input_json}'",
        f"out_dir='{output_dir}'",
        f"diffusion_batch_size={str(number_of_models)}",
        f"seed={str(seed)}",
        f"ckpt_path='{ckpt_path}'"
    ]


def generate_rosettafold_test_command() -> list:
    """
    Generate the test command for RosettaFold3

    Args:
        None

    Returns:
        list: The OpenFold 3 test command
    """

    return [
        "rf3",
        "fold",
        "--help",
    ]
