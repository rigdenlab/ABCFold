import os
import tempfile

import pytest

from abcfold.openfold3.run_openfold3 import (generate_openfold_command,
                                             run_openfold)


@pytest.mark.skipif(os.getenv("CI") == "true", reason="Skipping test in CI environment")
def test_run_openfold(test_data):

    with tempfile.TemporaryDirectory() as temp_dir:
        try:
            run_openfold(
                test_data.test_inputA_json,
                temp_dir,
                config=test_data.config_dict,
                save_input=True,
                test=True,
            )
        except Exception as e:
            print(e)
            assert False


def test_generate_openfold_command(test_data):
    input_json = "/road/to/nowhere.json"
    input_yaml = "/road/to/nowhere.yaml"
    output_dir = "/road/to/nowhere"
    ckpt_path = "/road/to/nowhere.ckpt"

    cmd = generate_openfold_command(
        input_json=input_json,
        output_dir=output_dir,
        runner_yaml=input_yaml,
        ckpt_path=ckpt_path,
        number_of_models=5,
    )

    assert "run_openfold" in cmd
    assert "predict" in cmd
    assert input_json in cmd
    assert input_yaml in cmd
    assert output_dir in cmd
    assert ckpt_path in cmd
    assert str(5) in cmd
