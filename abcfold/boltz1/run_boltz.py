import logging
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Union
import os

from abcfold.boltz1.af3_to_boltz1 import BoltzYaml
from abcfold.boltz1.check_install import check_boltz1

logger = logging.getLogger("logger")


def normalize_gpus(gpus: str) -> str:
    if gpus == "cpu":
        return ""
    if gpus == "all":
        return None
    gpu_ids = []
    for gpu in gpus.split(","):
        gpu = gpu.strip()
        if not gpu.isdigit():
            raise ValueError(f"Invalid GPU ID: {gpu}")
        gpu_ids.append(gpu)
    return ",".join(gpu_ids)


def run_boltz(
    input_json: Union[str, Path],
    output_dir: Union[str, Path],
    save_input: bool = False,
    test: bool = False,
    number_of_models: int = 5,
    num_recycles: int = 10,
    gpus: str = "all",
) -> bool:
    """
    Run Boltz1 using the input JSON file

    Args:
        input_json (Union[str, Path]): Path to the input JSON file
        output_dir (Union[str, Path]): Path to the output directory
        save_input (bool): If True, save the input yaml file and MSA to the output
        directory
        test (bool): If True, run the test command
        number_of_models (int): Number of models to generate

    Returns:
        Bool: True if the Boltz1 run was successful, False otherwise

    Raises:
        subprocess.CalledProcessError: If the Boltz1 command returns an error


    """
    input_json = Path(input_json)
    output_dir = Path(output_dir)

    logger.debug("Checking if boltz1 is installed")
    check_boltz1()

    with tempfile.TemporaryDirectory() as temp_dir:
        working_dir = Path(temp_dir)
        if save_input:
            logger.info("Saving input yaml file and msa to the output directory")
            working_dir = output_dir

        boltz_yaml = BoltzYaml(working_dir)
        boltz_yaml.json_to_yaml(input_json)
        out_file = working_dir.joinpath(f"{input_json.stem}.yaml")

        boltz_yaml.write_yaml(out_file)
        logger.info("Running Boltz1")
        cmd = (
            generate_boltz_command(out_file, output_dir, number_of_models, num_recycles, gpus)
            if not test
            else generate_boltz_test_command()
        )

        env = os.environ.copy()
        try:
            cuda_devices = normalize_gpus(gpus)
            if cuda_devices is not None:
                env["CUDA_VISIBLE_DEVICES"] = cuda_devices
        except ValueError as e:
            logger.error(f"Error setting up GPU: {e}")
            return False
        def run_cmd(cmd, env):
            with subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=env,
            ) as proc:
                stdout = ""
                if proc.stdout:
                    for line in proc.stdout:
                        sys.stdout.write(line.decode())
                        sys.stdout.flush()
                        stdout += line.decode()
                _, stderr = proc.communicate()
                return proc.returncode, stdout, stderr
        returncode, stdout, stderr = run_cmd(cmd, env)
        if returncode != 0 and b"Missing MSA's in input and --use_msa_server flag not set" in stderr:
            logger.warning("MSA not found, retrying with --use_msa_server")
            cmd.append("--use_msa_server")
            returncode, stdout, stderr = run_cmd(cmd, env)
        if returncode != 0:
            if stderr:
                logger.error(stderr.decode())
                output_err_file = output_dir / "boltz_error.log"
                with open(output_err_file, "w") as f:
                    f.write(stderr.decode())
                logger.error(
                    "Boltz1 run failed. Error log is in %s", output_err_file
                )
            else:
                logger.error("Boltz1 run failed")
            return False
        elif "WARNING: ran out of memory" in stdout:
            logger.error("Boltz1 ran out of memory")
            return False
        logger.info("Boltz1 run complete")
        logger.info("Output files are in %s", output_dir)
        return True


def generate_boltz_command(
    input_yaml: Union[str, Path],
    output_dir: Union[str, Path],
    number_of_models: int = 5,
    num_recycles: int = 10,
    gpus: str = "all",
) -> list:
    """
    Generate the Boltz1 command

    Args:
        input_yaml (Union[str, Path]): Path to the input YAML file
        output_dir (Union[str, Path]): Path to the output directory
        number_of_models (int): Number of models to generate

    Returns:
        list: The Boltz1 command
    """
    cmd = [
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
    ]
    return cmd


def generate_boltz_test_command() -> list:
    """
    Generate the test command for Boltz1

    Args:
        None

    Returns:
        list: The Boltz1 test command
    """

    return [
        "boltz",
        "predict",
        "--help",
    ]
