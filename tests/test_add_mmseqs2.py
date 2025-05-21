import json
import tempfile
from pathlib import Path

import pytest

from abcfold.scripts.add_mmseqs_msa import (MMseqs2Exception, add_msa_to_json,
                                            fetch_mmcif, run_mmseqs)


def test_add_msa_to_json(test_data):
    with open(test_data.test_inputA_json) as f:
        input_dict = json.load(f)
    add_msa_to_json(
        input_json=test_data.test_inputA_json,
        mmseqs_db=None,
        templates=test_data.test_6BJ9_cif,
        num_templates=20,
        chai_template_output=None,
        custom_template=None,
        custom_template_chain=None,
        target_id=None,
        input_params=None,
        output_json=None,
    )

    out_json = Path(test_data.test_inputA_json).parent / "inputA_mmseqs.json"

    assert out_json.exists()
    out_json.unlink()

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        add_msa_to_json(
            input_json=test_data.test_inputA_json,
            mmseqs_db=None,
            templates=None,
            num_templates=20,
            chai_template_output=None,
            custom_template=None,
            custom_template_chain=None,
            target_id=None,
            input_params=input_dict,
            output_json=tmpdir / "output.json",
        )

        assert (tmpdir / "output.json").exists()
        with open(tmpdir / "output.json") as f:
            output_dict = json.load(f)

        assert "unpairedMsa" in output_dict["sequences"][0]["protein"]
        assert "pairedMsa" in output_dict["sequences"][0]["protein"]
        assert "templates" in output_dict["sequences"][0]["protein"]
        with pytest.raises(ValueError):
            add_msa_to_json(
                input_json=test_data.test_inputAB_json,
                mmseqs_db=None,
                templates=None,
                num_templates=20,
                chai_template_output=None,
                custom_template=[test_data.test_6BJ9_cif],
                custom_template_chain=[None],
                target_id=None,
                input_params=input_dict,
                output_json=tmpdir / "output.json",
            )
        with pytest.raises(FileNotFoundError):
            add_msa_to_json(
                input_json=test_data.test_inputAB_json,
                mmseqs_db=None,
                templates=None,
                num_templates=20,
                chai_template_output=None,
                custom_template=["road/to/nowhere"],
                custom_template_chain=[None],
                target_id=None,
                input_params=input_dict,
                output_json=tmpdir / "output.json",
            )


def test_run_mmseqs(test_data):
    # Code is taken from ColabFold and not modified so no need to test
    # but not exhaustive
    with open(test_data.test_inputA_json) as f:
        input_dict = json.load(f)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        sequence = input_dict["sequences"][0]
        input_sequence = sequence["protein"]["sequence"]

        run_mmseqs(
            input_sequence,
            tmpdir,
            use_templates=True,
            num_templates=20,
        )


def test_mms2_exception():
    with pytest.raises(MMseqs2Exception):
        raise MMseqs2Exception()


def test_fetch_mmcif():
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        cif_str = fetch_mmcif("6BJ9", "A", 1, 100, tmpdir)
        assert cif_str is not None
        assert len(cif_str) > 0
        assert isinstance(cif_str, str)

        assert len(cif_str) == 110528
        cif_list = cif_str.split("\n")

        assert cif_list[0] == "data_6bj9"
        assert len(cif_list) == 1479
