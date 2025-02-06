import os
import subprocess
from pathlib import Path

import pytest

from abcfold.run_alphafold3 import generate_af3_cmd, run_alphafold3


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

    cmd = generate_af3_cmd(
        input_json=input_json,
        output_dir=output_dir,
        model_params=model_params,
        database_dir=database_dir,
        interactive=False,
    )
    assert "docker run" in cmd


@pytest.mark.skipif(os.getenv("CI") == "true", reason="Skipping test in CI environment")
def test_run_af3(test_data):
    input_json = Path(test_data.test_inputA_json)
    output_dir = Path("/road/to/nowhere")
    model_params = Path("/road/to/nowhere")
    database_dir = Path("/road/to/nowhere")

    with pytest.raises(subprocess.CalledProcessError):
        run_alphafold3(
            input_json,
            output_dir,
            model_params,
            database_dir,
            interactive=False,
        )

    cmd = generate_af3_cmd(
        input_json=input_json,
        output_dir=output_dir,
        model_params=model_params,
        database_dir=database_dir,
        interactive=False,
    )

    with subprocess.Popen(
        cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
    ) as p:
        _, stderr = p.communicate()
        assert p.returncode == 126
        print(stderr.decode())
        assert (
            "Error response from daemon: error while creating mount source path \
'/road/to/nowhere': mkdir /road: permission denied"
            in stderr.decode()
        )
