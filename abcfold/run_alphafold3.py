import logging
import subprocess
import sys
from pathlib import Path
from typing import Union

from packaging.version import Version

logger = logging.getLogger("logger")

AF3_VERSION = "3.0.0"


def check_af3_install(interactive: bool = True) -> None:
    """
    Check if Alphafold3 is installed by running the help command

    Args:
        interactive (bool): If True, run the docker container in interactive mode

    Raises:
        subprocess.CalledProcessError: If the Alphafold3 help command returns an error

    """
    logger.debug("Checking if Alphafold3 is installed")
    cmd = generate_test_command(interactive)
    with subprocess.Popen(
        cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
    ) as p:
        _, stderr = p.communicate()
        p.wait()
        if p.returncode != 1:
            logger.error(
                "Alphafold3 is not installed, please go to \
https://github.com/google-deepmind/alphafold3 and follow install instructions"
            )

            raise subprocess.CalledProcessError(p.returncode, cmd, stderr)
    logger.info("Alphafold3 is installed")

    cmd = generate_version_command()
    with subprocess.Popen(
        cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
    ) as p:
        stdout, stderr = p.communicate()
        version = str(stdout.strip().decode('utf-8'))

        if Version(version) < Version(AF3_VERSION):
            raise ImportError(
                f"Expected AlphaFold3 version {AF3_VERSION} or later, found {version}"
            )


def run_alphafold3(
    input_json: Union[str, Path],
    output_dir: Union[str, Path],
    model_params: Union[str, Path],
    database_dir: Union[str, Path],
    interactive: bool = False,
    number_of_models: int = 5,
    num_recycles: int = 10,
) -> None:
    """
    Run Alphafold3 using the input JSON file

    Args:
        input_json (Union[str, Path]): Path to the input JSON file
        output_dir (Union[str, Path]): Path to the output directory
        model_params (Union[str, Path]): Path to the model parameters
        database_dir (Union[str, Path]): Path to the database directory
        interactive (bool): If True, run the docker container in interactive mode
        number_of_models (int): Number of models to generate

    Returns:
        None

    Raises:
        subprocess.CalledProcessError: If the Alphafold3 command returns an error

    """

    check_af3_install(interactive)

    input_json = Path(input_json)
    output_dir = Path(output_dir)
    cmd = generate_af3_cmd(
        input_json=input_json,
        output_dir=output_dir,
        model_params=model_params,
        database_dir=database_dir,
        interactive=interactive,
        number_of_models=number_of_models,
        num_recycles=num_recycles,
    )

    logger.info("Running Alphafold3")
    with subprocess.Popen(
        cmd, shell=True, stdout=sys.stdout, stderr=subprocess.PIPE
    ) as p:
        _, stderr = p.communicate()
        if p.returncode != 0:
            logger.error(stderr.decode())
            raise subprocess.CalledProcessError(p.returncode, cmd, stderr)
    logger.info("Alphafold3 run complete")
    logger.info("Output files are in %s", output_dir)


def generate_af3_cmd(
    input_json: Union[str, Path],
    output_dir: Union[str, Path],
    model_params: Union[str, Path],
    database_dir: Union[str, Path],
    number_of_models: int = 10,
    num_recycles: int = 5,
    interactive: bool = False,
) -> str:
    """
    Generate the Alphafold3 command

    Args:
        input_json (Union[str, Path]): Path to the input JSON file
        output_dir (Union[str, Path]): Path to the output directory
        model_params (Union[str, Path]): Path to the model parameters
        database_dir (Union[str, Path]): Path to the database directory
        number_of_models (int): Number of models to generate
        interactive (bool): If True, run the docker container in interactive mode

    Returns:
        str: The Alphafold3 command
    """
    input_json = Path(input_json)
    output_dir = Path(output_dir)
    return f"""
    docker run {'-it' if interactive else ''} \
    --volume {input_json.parent.resolve()}:/root/af_input \
    --volume {output_dir.resolve()}:/root/af_output \
    --volume {model_params}:/root/models \
    --volume {database_dir}:/root/public_databases \
    --gpus all \
    alphafold3 \
    python run_alphafold.py \
    --json_path=/root/af_input/{input_json.name} \
    --model_dir=/root/models \
    --output_dir=/root/af_output \
    --num_diffusion_samples {number_of_models}\
    --num_recycles {num_recycles}
    """


def generate_test_command(interactive: bool = True) -> str:
    """
    Generate the Alphafold3 help command

    Args:
        interactive (bool): If True, run the docker container in interactive mode

    Returns:
        str: The Alphafold3 help command
    """
    return f"""
    docker run {'-it' if interactive else ''} \
    alphafold3 \
    python run_alphafold.py \
    --help
"""


def generate_version_command() -> str:
    """
    Generate the Alphafold3 version command
    """

    return """docker run \
    alphafold3 \
    python -c \
    'from alphafold3.version import __version__ ; print(__version__)'
"""
