from pathlib import Path
import subprocess
import logging

logger = logging.getLogger("logger")

def run_alphafold3(
    input_json: Path,
    output_dir: Path,
    model_params: Path,
    database_dir: Path,
    number_of_models: int = 5,
    num_recycles: int = 10,
    gpus: str = "all",
    interactive: bool = False,
) -> bool:

    cmd = _build_docker_cmd(
        input_json, output_dir, model_params, database_dir,
        number_of_models, num_recycles, gpus, interactive
    )

    logger.info("Running Alphafold3 → GPU=%s", gpus)
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        (output_dir / "af3_error.log").write_text(e.stderr.decode() if e.stderr else "")
        logger.error("Alphafold3 failed, see af3_error.log")
        return False

    return True


def _build_docker_cmd(
    input_json: Path,
    output_dir: Path,
    model_params: Path,
    database_dir: Path,
    n_models: int,
    n_recycles: int,
    gpus: str,
    interactive: bool,
) -> list[str]:

    cmd = ["docker", "run", "-i"] if interactive else ["docker", "run", "--rm"]


    if gpus.lower() == "cpu":
        pass
    elif gpus.lower() == "all":
        cmd += ["--gpus", "all"]
    else:
        cmd += ["--gpus", f"device={gpus}"]


    cmd += [
        "--volume", f"{input_json.parent.resolve()}:/root/af_input:ro",
        "--volume", f"{output_dir.resolve()}:/root/af_output",
        "--volume", f"{model_params}:/root/models:ro",
        "--volume", f"{database_dir}:/root/public_databases:ro",
        "alphafold3",
        "python", "run_alphafold.py",
        "--json_path", f"/root/af_input/{input_json.name}",
        "--model_dir", "/root/models",
        "--output_dir", "/root/af_output",
        "--num_diffusion_samples", str(n_models),
        "--num_recycles", str(n_recycles),
    ]
    return cmd
