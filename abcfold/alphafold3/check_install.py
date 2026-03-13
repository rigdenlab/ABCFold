import logging
import subprocess
from pathlib import Path
from typing import Union

from packaging.version import Version

logger = logging.getLogger(__name__)


def check_af3_install(config: dict,
                      interactive: bool = True,
                      sif_path: Union[str, Path, None] = None) -> None:
    """
    Check if Alphafold3 is installed by running the help command

    Args:
        config (dict): Configuration dictionary
        interactive (bool): If True, run the docker container in interactive mode
        sif_path (Union[str, Path, None]): Path to a Singularity image file

    Raises:
        subprocess.CalledProcessError: If the Alphafold3 help command returns an error

    """
    logger.debug("Checking if Alphafold3 is installed")
    AF3_VERSION = config["af3_version"]
    cmd = generate_test_command(config, interactive, sif_path)
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

    cmd = generate_version_command(config, sif_path)
    with subprocess.Popen(
        cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
    ) as p:
        stdout, stderr = p.communicate()
        version = str(stdout.strip().decode("utf-8"))

        if Version(version) < Version(AF3_VERSION):
            raise ImportError(
                f"Expected AlphaFold3 version {AF3_VERSION} or later, found {version}"
            )


def generate_test_command(config: dict,
                          interactive: bool = True,
                          sif_path: Union[str, Path, None] = None,
                          ) -> str:
    """
    Generate the Alphafold3 help command

    Args:
        config (dict): Configuration dictionary
        interactive (bool): If True, run the docker container in interactive mode
        sif_path (Union[str, Path, None]): Path to a Singularity image file

    Returns:
        str: The Alphafold3 help command
    """

    if sif_path is not None and sif_path != "None":
        return f"""
    singularity exec \
    {sif_path} \
    python /app/alphafold/run_alphafold.py \
    --help
"""
    else:
        return f"""
    docker run {'-it' if interactive else ''} \
    {config["af3_docker_env"]} \
    python run_alphafold.py \
    --help
"""


def generate_version_command(
        config: dict,
        sif_path: Union[str, Path, None] = None) -> str:
    """
    Generate the Alphafold3 version command
    """
    if sif_path is not None and sif_path != "None":
        return f"""
    singularity exec \
    {sif_path} \
    python -c \
    'from alphafold3.version import __version__ ; print(__version__)'
"""
    else:
        return f"""
    docker run \
    {config["af3_docker_env"]} \
    python -c \
    'from alphafold3.version import __version__ ; print(__version__)'
"""
