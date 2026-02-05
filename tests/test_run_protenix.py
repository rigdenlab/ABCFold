import os
import tempfile
from pathlib import Path

import pytest

from abcfold.protenix.af3_to_protenix import ProtenixJson
from abcfold.protenix.run_protenix import (generate_protenix_command,
                                           run_protenix)


@pytest.mark.skipif(os.getenv("CI") == "true", reason="Skipping test in CI environment")
def test_run_protenix(test_data):

    with tempfile.TemporaryDirectory() as temp_dir:
        try:
            run_protenix(
                test_data.test_inputA_json,
                temp_dir,
                config=test_data.config_dict,
                save_input=True,
                test=True,
            )
        except Exception as e:
            print(e)
            assert False


def test_generate_protenix_command(test_data):
    with tempfile.TemporaryDirectory() as temp_dir:

        # Need protenix style json file to check whether to use MSA flag
        protenix_json = ProtenixJson(temp_dir)
        protenix_json.json_to_json(test_data.test_inputAmsa_json)
        protenix_json_path = Path(temp_dir) / "protenix1.json"
        protenix_json.write_json(protenix_json_path)

        output_dir = "/road/to/nowhere"

        cmd = generate_protenix_command(
            input_json=protenix_json_path,
            output_dir=output_dir,
            number_of_models=5,
            num_recycles=3,
            seed=42
        )

        assert "runner.inference" in cmd
        assert "--input_json_path" in cmd
        assert protenix_json_path.as_posix() in cmd
        assert "--dump_dir" in cmd
        assert output_dir in cmd
        assert "--sample_diffusion.N_sample" in cmd
        assert "5" in cmd
        assert "--model.N_cycle" in cmd
        assert "3" in cmd
        assert "--seeds" in cmd
        assert "42" in cmd
        assert "--use_msa" in cmd
        assert "True" in cmd
