import os
import tempfile

import pytest

try:
    import chai_lab  # noqa F401

    run_chai1 = True

except ImportError:
    run_chai1 = False


@pytest.mark.skipif(not run_chai1, reason="chai_lab not installed")
def test_generate_chai_command(test_data):
    from abcfold.chai1.run_chai1 import generate_chai_command

    input_fasta = "/road/to/nowhere.fasta"
    msa_dir = "/road/to/nowhere"
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as fp:
        constraints = fp.name
        output_dir = "/road/to/nowhere"

        cmd = generate_chai_command(
            input_fasta=input_fasta,
            msa_dir=msa_dir,
            input_constraints=constraints,
            output_dir=output_dir,
        )

    assert cmd[1].endswith("chai.py")
    assert "fold" in cmd
    assert input_fasta in cmd
    assert msa_dir in cmd
    assert constraints in cmd
    assert output_dir in cmd
    assert "--num-diffn-samples" in cmd
    assert "5" in cmd


@pytest.mark.skipif(
    os.getenv("CI") == "true" and not run_chai1,
    reason="Skipping test in CI environment",
)
def test_run_chai(test_data):
    pytest.importorskip("chai_lab")
    from abcfold.chai1.run_chai1 import run_chai

    with tempfile.TemporaryDirectory() as temp_dir:
        try:
            run_chai(
                test_data.test_inputA_json,
                temp_dir,
                save_input=True,
                test=True,
                config=test_data.config_dict,
            )
        except Exception as e:
            print(e)
            assert False
