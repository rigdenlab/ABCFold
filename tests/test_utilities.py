from pathlib import Path

from abcfold.abc_script_utils import (align_and_map, check_input_json,
                                      extract_sequence_from_mmcif, get_chains,
                                      get_mmcif)
from abcfold.processoutput.file_handlers import CifFile


def test_get_chains(test_data):
    chains = get_chains(test_data.test_6BJ9_cif)
    assert all(isinstance(chain, str) for chain in chains)
    assert len(chains) == 2
    assert "A" in chains
    assert "B" in chains


def test_extract_sequence_from_mmcif(test_data):
    with open(test_data.test_6BJ9_cif) as f:
        sequence = extract_sequence_from_mmcif(f)
    assert isinstance(sequence, str)
    assert len(sequence) == 770
    assert sequence.startswith("A")
    assert sequence.endswith("L")
    assert sequence.count("A") == 178
    assert " " not in sequence


def test_align_and_map(test_data):
    with open(test_data.test_6BJ9_cif) as f:
        template_seq = extract_sequence_from_mmcif(f)
    query_seq = template_seq[::-1]
    query_indices, template_indices = align_and_map(query_seq, template_seq)
    assert len(query_indices) == 385
    assert len(template_indices) == 385
    assert all(isinstance(i, int) for i in query_indices)
    assert all(isinstance(i, int) for i in template_indices)
    assert query_indices[0] == 3
    assert template_indices[0] == 2


def test_get_mmcif(test_data):
    cif_str = get_mmcif(test_data.test_6BJ9_cif, "6BJ9", "A", 1, 100)
    assert isinstance(cif_str, str)
    assert len(cif_str) == 110528
    cif_list = cif_str.split("\n")

    assert cif_list[0] == "data_6BJ9"
    assert len(cif_list) == 1479

    cif_str = get_mmcif(test_data.test_6BJ9_metadata_strip_cif, "6BJ9", "A", 1, 100)

    assert "_pdbx_audit_revision_history.revision_date" in cif_str

    cif_str = get_mmcif(test_data.test_1G03_cif, "1G03", "A", 1, 100)
    assert cif_str.startswith("data_1G03")


def test_check_input_json(test_data):
    input_json = test_data.test_inputAmsa_json
    input_json1 = check_input_json(input_json, use_af3_templates=True, test=True)
    input_json2 = check_input_json(input_json, use_af3_templates=False, test=True)

    assert "templates" in input_json1["sequences"][0]["protein"]
    assert input_json1["sequences"][0]["protein"]["templates"] is None

    assert "templates" in input_json2["sequences"][0]["protein"]
    assert input_json2["sequences"][0]["protein"]["templates"] == []


def test_clash_checker(test_data):
    cif_file = Path(test_data.test_boltz_1_6BJ9_).joinpath(
        "predictions", "test_mmseqs", "test_mmseqs_model_0.cif"
    )

    structure = CifFile(cif_file)
    clashes = structure.check_clashes()
    assert isinstance(clashes, list)
    assert len(clashes) == 2
