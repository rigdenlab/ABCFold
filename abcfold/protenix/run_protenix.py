import json
import logging
import subprocess
import tempfile
from pathlib import Path
from typing import Optional, Union

from abcfold.protenix.af3_to_protenix import ProtenixJson
from abcfold.protenix.check_install import (ensure_protenix_checkpoint,
                                            ensure_protenix_env)

logger = logging.getLogger("logger")


def run_protenix(
    input_json: Union[str, Path],
    output_dir: Union[str, Path],
    config: dict,
    save_input: bool = False,
    test: bool = False,
    number_of_models: int = 5,
    num_recycles: int = 10,
) -> bool:
    """
    Run Protenix using the input JSON file

    Args:
        input_json (Union[str, Path]): Path to the input JSON file
        output_dir (Union[str, Path]): Path to the output directory
        config (dict): Configuration dictionary
        save_input (bool): If True, save the input yaml file and MSA to the output
        directory
        test (bool): If True, run the test command
        number_of_models (int): Number of models to generate
        num_recycles (int): Number of recycles

    Returns:
        Bool: True if the Protenix run was successful, False otherwise

    Raises:
        subprocess.CalledProcessError: If the Protenix command returns an error


    """
    input_json = Path(input_json)
    output_dir = Path(output_dir)

    logger.debug("Checking if protenix is installed")
    env = ensure_protenix_env(config=config)

    protenix_weight_dir = config.get("protenix_weights")
    if protenix_weight_dir is not None and protenix_weight_dir != "None":
        checkpoint_dir = Path(protenix_weight_dir)
    else:
        checkpoint_dir = Path.home().joinpath("checkpoint")

    try:
        ensure_protenix_checkpoint(checkpoint_dir, config["protenix_model"])
    except RuntimeError as e:
        logger.error(str(e))
        return False

    with tempfile.TemporaryDirectory() as temp_dir:
        working_dir = Path(temp_dir)
        if save_input:
            logger.info("Saving input json file and msa to the output directory")
            working_dir = output_dir

        protenix_json = ProtenixJson(working_dir)
        protenix_json.json_to_json(input_json)

        for seed in protenix_json.seeds:
            out_file = working_dir.joinpath(f"{input_json.stem}_seed-{seed}.json")

            protenix_json.write_json(out_file)
            protenix_out_dir = output_dir / f"protenix_results_seed-{seed}"
            logger.info("Running Protenix using seed: %s", seed)
            cmd = (
                generate_protenix_command(
                    out_file,
                    protenix_out_dir,
                    config,
                    number_of_models,
                    num_recycles,
                    seed=seed,
                    checkpoint_dir=checkpoint_dir,
                )
                if not test
                else generate_protenix_test_command()
            )

            try:
                env.run(cmd)
            except subprocess.CalledProcessError as e:
                stderr = e.stderr or ""
                if stderr:
                    if working_dir.exists():
                        output_err_file = working_dir / "protenix_error.log"
                    else:
                        output_err_file = working_dir.parent / "protenix_error.log"
                    output_err_file.write_text(stderr)
                    logger.error(
                        "Protenix run failed. Error log is in %s", output_err_file
                    )
                else:
                    logger.error("Protenix run failed")
                return False

    logger.info("Protenix run complete")
    logger.info("Output files are in %s", output_dir)
    return True


def generate_protenix_command(
    input_json: Union[str, Path],
    output_dir: Union[str, Path],
    config: dict,
    number_of_models: int,
    num_recycles: int,
    seed: int,
    checkpoint_dir: Optional[Union[str, Path]] = None,
) -> list:
    """
    Generate the Protenix command

    Args:
        input_json (Union[str, Path]): Path to the input JSON file
        output_dir (Union[str, Path]): Path to the output directory
        config (dict): Configuration dictionary
        number_of_models (int): Number of models to generate
        num_recycles (int): Number of recycles
        seed (int): Random seed
        config (dict): Configuration dictionary
        checkpoint_dir (Union[str, Path]): Directory containing the Protenix
            checkpoint, passed through as --load_checkpoint_dir. If not given,
            protenix falls back to its own default (~/checkpoint).

    Returns:
        list: The Protenix command
    """

    # Determine if MSA is present in the input JSON
    use_msa = False
    with open(str(input_json), "r") as f:
        data = json.load(f)
    for key, value in data[0].items():
        if key == "sequences":
            for entry in value:
                if "proteinChain" in entry:
                    if "msa" in entry["proteinChain"]:
                        use_msa = True
                        break

    cmd = [
        "python",
        "-m",
        "runner.inference",
        "--model_name",
        str(config["protenix_model"]),
        "--input_json_path",
        str(input_json),
        "--dump_dir",
        str(output_dir),
        "--sample_diffusion.N_sample",
        str(number_of_models),
        "--model.N_cycle",
        str(num_recycles),
        "--seeds",
        str(seed),
        "--use_msa",
        str(use_msa),
        "--need_atom_confidence",
        "True"
    ]

    if checkpoint_dir is not None:
        cmd += ["--load_checkpoint_dir", str(checkpoint_dir)]

    return cmd


def generate_protenix_test_command() -> list:
    """
    Generate the test command for Protenix

    Args:
        None

    Returns:
        list: The Protenix test command
    """

    return [
        "protenix",
        "pred",
        "--help",
    ]
