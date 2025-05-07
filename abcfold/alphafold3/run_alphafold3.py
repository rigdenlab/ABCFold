import logging
import subprocess
import sys
from pathlib import Path
from typing import Union

logger = logging.getLogger("logger")


def run_alphafold3(
    input_json: Union[str, Path],
    output_dir: Union[str, Path],
    model_params: Union[str, Path],
    database_dir: Union[str, Path],
    sif_path: Union[str, Path, None],
    interactive: bool = False,
    number_of_models: int = 5,
    num_recycles: int = 10,
) -> bool:
    """
    Run Alphafold3 using the input JSON file

    Args:
        input_json (Union[str, Path]): Path to the input JSON file
        output_dir (Union[str, Path]): Path to the output directory
        model_params (Union[str, Path]): Path to the model parameters
        database_dir (Union[str, Path]): Path to the database directory
        sif_path (Union[str, Path, None]): Path to a Singularity image file
        interactive (bool): If True, run the docker container in interactive mode
        number_of_models (int): Number of models to generate

    Returns:
        Bool: True if the Alphafold3 run was successful, False otherwise

    Raises:
        subprocess.CalledProcessError: If the Alphafold3 command returns an error

    """

    input_json = Path(input_json)
    output_dir = Path(output_dir)
    cmd = generate_af3_cmd(
        input_json=input_json,
        output_dir=output_dir,
        model_params=model_params,
        database_dir=database_dir,
        sif_path=sif_path,
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
            output_err_file = output_dir / "af3_error.log"
            with open(output_err_file, "w") as f:
                f.write(stderr.decode())
            logger.error("Alphafold3 run failed. Error log is in %s", output_err_file)
            return False

    logger.info("Alphafold3 run complete")
    logger.info("Output files are in %s", output_dir)
    return True


def generate_af3_cmd(
    input_json: Union[str, Path],
    output_dir: Union[str, Path],
    model_params: Union[str, Path],
    database_dir: Union[str, Path],
    sif_path: Union[str, Path, None],
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
        sif_path (Union[str, Path, None]): Path to a Singularity image file
        number_of_models (int): Number of models to generate
        interactive (bool): If True, run the docker container in interactive mode

    Returns:
        str: The Alphafold3 command
    """
    input_json = Path(input_json)
    output_dir = Path(output_dir)

    if sif_path is not None:
        return f"""
        singularity exec \
        --nv \
        --bind {input_json.parent.resolve()}:/root/af_input \
        --bind {output_dir.resolve()}:/root/af_output \
        --bind {model_params}:/root/models \
        --bind {database_dir}:/root/public_databases \
        {sif_path} \
        python /app/alphafold/run_alphafold.py \
        --json_path=/root/af_input/{input_json.name} \
        --model_dir=/root/models \
        --output_dir=/root/af_output \
        --num_diffusion_samples {number_of_models}\
        --num_recycles {num_recycles}
    """

    else:
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
