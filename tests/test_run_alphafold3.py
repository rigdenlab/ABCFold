import os
import tempfile
from pathlib import Path

import pytest

from abcfold.alphafold3.run_alphafold3 import generate_af3_cmd, run_alphafold3
from abcfold.scripts.abc_script_utils import make_dummy_af3_db


def test_generate_af3_command(test_data):
    input_json = Path(test_data.test_inputA_json)
    output_dir = Path("/road/to/nowhere")
    model_params = Path("/road/to/nowhere")
    database_dir = Path("/road/to/nowhere")

    cmd = generate_af3_cmd(
        input_json=input_json,
        output_dir=output_dir,
        model_params=model_params,
        database_dir=database_dir,
        sif_path=None,
        interactive=True,
    )

    assert "docker run -it" in cmd
    assert f"--volume {input_json.parent.resolve()}:/root/af_input" in cmd
    assert f"--volume {output_dir.resolve()}:/root/af_output" in cmd
    assert f"--volume {model_params}:/root/models" in cmd
    assert f"--volume {database_dir}:/root/public_databases" in cmd
    assert "--gpus all" in cmd
    assert "alphafold3" in cmd
    assert f"--json_path=/root/af_input/{input_json.name}" in cmd
    assert "--model_dir=/root/models" in cmd
    assert "--output_dir=/root/af_output" in cmd


def test_generate_af3_singularity_command(test_data):
    input_json = Path(test_data.test_inputA_json)
    output_dir = Path("/road/to/nowhere")
    model_params = Path("/road/to/nowhere")
    database_dir = Path("/road/to/nowhere")
    sif_path = Path("/road/to/nowhere.sif")

    cmd = generate_af3_cmd(
        input_json=input_json,
        output_dir=output_dir,
        model_params=model_params,
        database_dir=database_dir,
        sif_path=sif_path,
        interactive=True,
    )

    assert "singularity exec" in cmd
    assert f"--bind {input_json.parent.resolve()}:/root/af_input" in cmd
    assert f"--bind {output_dir.resolve()}:/root/af_output" in cmd
    assert f"--bind {model_params}:/root/models" in cmd
    assert f"--bind {database_dir}:/root/public_databases" in cmd
    assert f"{sif_path}" in cmd
    assert "python /app/alphafold/run_alphafold.py" in cmd
    assert f"--json_path=/root/af_input/{input_json.name}" in cmd
    assert "--model_dir=/root/models" in cmd
    assert "--output_dir=/root/af_output" in cmd
    assert "--num_diffusion_samples" in cmd
    assert "--num_recycles" in cmd


@pytest.mark.skipif(os.getenv("CI") == "true", reason="Skipping test in CI environment")
def test_run_af3(test_data):
    input_json = Path(test_data.test_inputA_json)
    with tempfile.TemporaryDirectory() as temp_dir_str:
        temp_dir = Path(temp_dir_str)
        output_dir = temp_dir / "af3_output"
        model_params = temp_dir / "af3_output"
        database_dir = make_dummy_af3_db(temp_dir)

        run_alphafold3(
            input_json,
            output_dir,
            model_params,
            database_dir,
            sif_path=None,
            interactive=False,
        )
        assert output_dir.exists()
        assert (output_dir / "af3_error.log").exists()
        with open(output_dir / "af3_error.log", "r") as f:
            error_log = f.read()
            assert "No models matched in /root/models" in error_log
