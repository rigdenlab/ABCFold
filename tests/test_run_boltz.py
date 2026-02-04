import os
import tempfile

import pytest

from abcfold.boltz.run_boltz import generate_boltz_command, run_boltz


@pytest.mark.skipif(os.getenv("CI") == "true", reason="Skipping test in CI environment")
def test_run_boltz(test_data):

    with tempfile.TemporaryDirectory() as temp_dir:
        try:
            run_boltz(
                test_data.test_inputA_json,
                temp_dir,
                config=test_data.config_dict,
                save_input=True,
                test=True,
            )
        except Exception as e:
            print(e)
            assert False


def test_generate_boltz_command(test_data):
    input_yaml = "/road/to/nowhere.yaml"
    output_dir = "/road/to/nowhere"

    cmd = generate_boltz_command(
        input_yaml=input_yaml,
        output_dir=output_dir,
        config=test_data.config_dict,
    )

    assert "boltz" in cmd
    assert "predict" in cmd
    assert input_yaml in cmd
    assert output_dir in cmd
    assert "--override" in cmd
