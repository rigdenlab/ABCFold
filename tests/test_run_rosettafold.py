import os
import tempfile

import pytest

from abcfold.rosettafold3.run_rosettafold3 import (
    generate_rosettafold_command, run_rosettafold)


@pytest.mark.skipif(os.getenv("CI") == "true", reason="Skipping test in CI environment")
def test_run_rosettafold(test_data):

    with tempfile.TemporaryDirectory() as temp_dir:
        try:
            run_rosettafold(
                test_data.test_inputA_json,
                temp_dir,
                config=test_data.config_dict,
                save_input=True,
                test=True,
            )
        except Exception as e:
            print(e)
            assert False


def test_generate_rosettafold_command(test_data):
    input_json = "/road/to/nowhere.json"
    output_dir = "/road/to/nowhere"
    ckpt_path = "/road/to/nowhere.ckpt"

    cmd = generate_rosettafold_command(
        input_json=input_json,
        output_dir=output_dir,
        ckpt_path=ckpt_path,
        number_of_models=5,
        seed=42
    )

    assert "rf3" in cmd
    assert "fold" in cmd
    assert f"inputs='{input_json}'" in cmd
    assert f"out_dir='{output_dir}'" in cmd
    assert f"ckpt_path='{ckpt_path}'" in cmd
    assert f"diffusion_batch_size={str(5)}" in cmd
    assert f"seed={str(42)}" in cmd
