import logging
import subprocess
import tempfile
from pathlib import Path
from typing import Union

from abcfold.boltz.af3_to_boltz import BoltzYaml
from abcfold.boltz.check_install import ensure_boltz_env

logger = logging.getLogger("logger")


def run_boltz(
    input_json: Union[str, Path],
    output_dir: Union[str, Path],
    config: dict,
    save_input: bool = False,
    test: bool = False,
    number_of_models: int = 5,
    num_recycles: int = 10,
) -> bool:
    """
    Run Boltz using the input JSON file

    Args:
        input_json (Union[str, Path]): Path to the input JSON file
        output_dir (Union[str, Path]): Path to the output directory
        config (dict): Configuration dictionary
        save_input (bool): If True, save the input yaml file and MSA to the output
        directory
        test (bool): If True, run the test command
        number_of_models (int): Number of models to generate
        num_recycles (int): Number of recycles to use

    Returns:
        Bool: True if the Boltz run was successful, False otherwise

    Raises:
        subprocess.CalledProcessError: If the Boltz command returns an error


    """
    input_json = Path(input_json)
    output_dir = Path(output_dir)

    logger.debug("Checking if boltz is installed")
    env = ensure_boltz_env(config=config)

    with tempfile.TemporaryDirectory() as temp_dir:
        working_dir = Path(temp_dir)
        if save_input:
            logger.info("Saving input yaml file and msa to the output directory")
            working_dir = output_dir

        boltz_yaml = BoltzYaml(working_dir)
        boltz_yaml.json_to_yaml(input_json)

        for seed in boltz_yaml.seeds:
            out_file = working_dir.joinpath(f"{input_json.stem}_seed-{seed}.yaml")

            boltz_yaml.write_yaml(out_file)
            logger.info("Running Boltz using seed: %s", seed)
            cmd = (
                generate_boltz_command(
                    out_file,
                    output_dir,
                    config,
                    number_of_models,
                    num_recycles,
                    seed=seed,
                )
                if not test
                else generate_boltz_test_command()
            )

            try:
                stdout = env.run(cmd, capture_output=True)

                # Check for out-of-memory warnings
                if stdout and "WARNING: ran out of memory" in stdout:
                    logger.error("Boltz ran out of memory")
                    return False

            except subprocess.CalledProcessError as e:
                stderr = e.stderr or ""
                if stderr:
                    logger.error(stderr)
                    output_err_file = output_dir / "boltz_error.log"
                    output_err_file.write_text(stderr)
                    logger.error(
                        "Boltz run failed. Error log is in %s", output_err_file
                    )
                else:
                    logger.error("Boltz run failed")
                return False

    logger.info("Boltz run complete")
    logger.info("Output files are in %s", output_dir)
    return True


def generate_boltz_command(
    input_yaml: Union[str, Path],
    output_dir: Union[str, Path],
    config: dict,
    number_of_models: int = 5,
    num_recycles: int = 10,
    seed: int = 42,
) -> list:
    """
    Generate the Boltz command

    Args:
        input_yaml (Union[str, Path]): Path to the input YAML file
        output_dir (Union[str, Path]): Path to the output directory
        config (dict): Configuration dictionary
        number_of_models (int): Number of models to generate
        seed (int): Seed for the random number generator

    Returns:
        list: The Boltz command
    """

    boltz_weight_dir = config['Weights']['boltz_weights']
    if boltz_weight_dir is not None and boltz_weight_dir != "None":
        cache_path = boltz_weight_dir
    else:
        cache_path = str(Path.home().joinpath(".boltz"))

    return [
        "boltz",
        "predict",
        str(input_yaml),
        "--out_dir",
        str(output_dir),
        "--override",
        "--write_full_pae",
        "--write_full_pde",
        "--diffusion_samples",
        str(number_of_models),
        "--recycling_steps",
        str(num_recycles),
        "--seed",
        str(seed),
        "--cache_path",
        str(cache_path),
    ]


def generate_boltz_test_command() -> list:
    """
    Generate the test command for Boltz

    Args:
        None

    Returns:
        list: The Boltz test command
    """

    return [
        "boltz",
        "predict",
        "--help",
    ]
