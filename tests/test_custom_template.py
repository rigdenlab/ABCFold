import json
import shutil
import tempfile
from pathlib import Path

import pytest

from abcfold.abc_script_utils import get_custom_template
from abcfold.add_custom_template import main, run_custom_template


def test_get_custom_template(test_data):
    with open(test_data.test_inputA_json) as f:
        input_dict = json.load(f)
    get_custom_template(
        input_dict["sequences"][0],
        "A",
        test_data.test_6BJ9_cif,
        "B",
    )
    assert "templates" in input_dict["sequences"][0]["protein"]
    assert "queryIndices" in input_dict["sequences"][0]["protein"]["templates"][0]
    assert "templateIndices" in input_dict["sequences"][0]["protein"]["templates"][0]

    with open(test_data.test_inputA_json) as f:
        input_dict = json.load(f)

    get_custom_template(
        input_dict["sequences"][0],
        "B",
        test_data.test_6BJ9_cif,
        "A",
    )
    assert "templates" not in input_dict["sequences"][0]["protein"]

    with open(test_data.test_inputA_template_json) as f:
        input_dict = json.load(f)

    get_custom_template(
        input_dict["sequences"][0],
        "A",
        test_data.test_6BJ9_cif,
        "B",
    )
    assert "templates" in input_dict["sequences"][0]["protein"]

    get_custom_template(
        input_dict["sequences"][0],
        "A",
        test_data.test_6BJ9A_cif,
        None,
    )
    assert "templates" in input_dict["sequences"][0]["protein"]

    with open(test_data.test_inputAB_json) as f:
        input_dict = json.load(f)

    get_custom_template(
        input_dict["sequences"][0],
        "C",
        test_data.test_6BJ9_cif,
        "B",
    )

    assert "templates" not in input_dict["sequences"][0]["protein"]


def test_get_custom_template_errors(test_data):
    with open(test_data.test_inputA_json) as f:
        input_dict = json.load(f)

    with pytest.raises(FileNotFoundError):
        get_custom_template(
            input_dict["sequences"][0],
            "A",
            "road/to/nowhere",
            "C",
        )

    with pytest.raises(ValueError):
        get_custom_template(
            input_dict["sequences"][0],
            "A",
            test_data.test_6BJ9_cif,
            None,
        )

    with pytest.raises(ValueError):
        get_custom_template(
            input_dict["sequences"][0],
            "A",
            test_data.test_6BJ9_cif,
            "C",
        )


def test_run_custom_template(test_data):
    # run main function
    with pytest.raises(SystemExit):
        main()

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        output_json = tmpdir / "output.json"
        run_custom_template(
            test_data.test_inputA_json,
            "A",
            test_data.test_6BJ9_cif,
            "B",
            output_json=output_json,
            to_file=True,
        )
        shutil.copyfile(test_data.test_inputA_json, tmpdir / "input.json")

        run_custom_template(
            tmpdir / "input.json",
            "A",
            test_data.test_6BJ9_cif,
            "B",
            output_json=None,
            to_file=True,
        )

        run_custom_template(
            test_data.test_inputRNA_json,
            "A",
            test_data.test_6BJ9_cif,
            "B",
            output_json=output_json,
            to_file=True,
        )

    with pytest.raises(FileNotFoundError):
        run_custom_template(
            test_data.test_inputA_json,
            "A",
            "road/to/nowhere",
            "C",
            output_json=None,
            to_file=False,
        )
