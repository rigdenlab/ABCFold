import json
import logging
import subprocess
import sys
from pathlib import Path
from typing import Union

from abcfold.alphafold3.ccd_one_letter import CCD_NAME_TO_ONE_LETTER
from abcfold.alphafold3.check_install import check_af3_install

logger = logging.getLogger("logger")


def run_alphafold3(
    input_json: Union[str, Path],
    output_dir: Union[str, Path],
    model_params: Union[str, Path],
    database_dir: Union[str, Path],
    sif_path: Union[str, Path, None],
    config: dict,
    interactive: bool = False,
    number_of_models: int = 5,
    num_recycles: int = 10,
    save_distogram: bool = False,
) -> bool:
    """
    Run Alphafold3 using the input JSON file

    Args:
        input_json (Union[str, Path]): Path to the input JSON file
        output_dir (Union[str, Path]): Path to the output directory
        model_params (Union[str, Path]): Path to the model parameters
        database_dir (Union[str, Path]): Path to the database directory
        sif_path (Union[str, Path, None]): Path to a Singularity image file
        config (dict): Configuration dictionary
        interactive (bool): If True, run the docker container in interactive mode
        number_of_models (int): Number of models to generate
        num_recycles (int): Number of recycles to use
        save_distogram (bool): If True, save the distogram output

    Returns:
        Bool: True if the Alphafold3 run was successful, False otherwise

    Raises:
        subprocess.CalledProcessError: If the Alphafold3 command returns an error

    """

    input_json = process_input_json(Path(input_json))
    output_dir = Path(output_dir)

    check_af3_install(config=config, interactive=False, sif_path=sif_path)

    cmd = generate_af3_cmd(
        input_json=input_json,
        output_dir=output_dir,
        model_params=model_params,
        database_dir=database_dir,
        sif_path=sif_path,
        config=config,
        interactive=interactive,
        number_of_models=number_of_models,
        num_recycles=num_recycles,
        save_distogram=save_distogram,
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


def process_input_json(input_json: Union[str, Path]) -> Union[str, Path]:
    """
    Process the input JSON file to post translational modifications (PTMs) are included
    in the MSA sequence

    Args:
        input_json (Union[str, Path]): Path to the input JSON file

    Returns:
        Union[str, Path]: Path to the processed input JSON file
    """

    with open(input_json, "r") as f:
        json_dict = json.load(f)

    one_letter_code = None
    position = None
    msa = None
    for sequence in json_dict['sequences']:
        protein = sequence.get("protein")
        if protein is not None:
            modifications = protein.get("modifications", [])
            for modification in modifications:
                if 'ptmType' in modification.keys():
                    ptm_type = modification['ptmType']
                    if ptm_type in CCD_NAME_TO_ONE_LETTER:
                        one_letter_code = CCD_NAME_TO_ONE_LETTER[ptm_type]
                        position = modification['ptmPosition']
                        msa = protein.get("unpairedMsa")
                if (
                    one_letter_code is not None
                    and position is not None
                    and msa is not None
                ):
                    msa_lines = msa.splitlines()
                    input_seq = msa_lines[1]
                    idx = int(position) - 1
                    input_seq = input_seq[:idx] + one_letter_code + input_seq[idx+1:]
                    msa_lines[1] = input_seq
                    protein['unpairedMsa'] = "\n".join(msa_lines)

    # Write the updated JSON dict back to the file
    with open(input_json, "w") as f:
        json.dump(json_dict, f, indent=4)

    return input_json


def generate_af3_cmd(
    input_json: Union[str, Path],
    output_dir: Union[str, Path],
    model_params: Union[str, Path],
    database_dir: Union[str, Path],
    sif_path: Union[str, Path, None],
    config: dict,
    number_of_models: int = 10,
    num_recycles: int = 5,
    interactive: bool = False,
    save_distogram: bool = False,
) -> str:
    """
    Generate the Alphafold3 command

    Args:
        input_json (Union[str, Path]): Path to the input JSON file
        output_dir (Union[str, Path]): Path to the output directory
        model_params (Union[str, Path]): Path to the model parameters
        database_dir (Union[str, Path]): Path to the database directory
        sif_path (Union[str, Path, None]): Path to a Singularity image file
        config (dict): Configuration dictionary
        number_of_models (int): Number of models to generate
        interactive (bool): If True, run the docker container in interactive mode
        num_recycles (int): Number of recycles to use
        save_distogram (bool): If True, save the distogram output

    Returns:
        str: The Alphafold3 command
    """
    input_json = Path(input_json)
    output_dir = Path(output_dir)

    if sif_path is not None and sif_path != "None":
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
        --db_dir=/root/public_databases \
        --num_diffusion_samples {number_of_models}\
        --num_recycles {num_recycles}\
        --save_distogram {str(save_distogram).lower()}
    """

    else:
        return f"""
        docker run {'-it' if interactive else ''} \
        --volume {input_json.parent.resolve()}:/root/af_input \
        --volume {output_dir.resolve()}:/root/af_output \
        --volume {model_params}:/root/models \
        --volume {database_dir}:/root/public_databases \
        --gpus all \
        {config["af3_docker_env"]} \
        python run_alphafold.py \
        --json_path=/root/af_input/{input_json.name} \
        --model_dir=/root/models \
        --output_dir=/root/af_output \
        --num_diffusion_samples {number_of_models}\
        --num_recycles {num_recycles}\
        --save_distogram {str(save_distogram).lower()}
        """
