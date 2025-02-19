from pathlib import Path

import pytest

from abcfold.plots.plotter import get_plddt_regions
from abcfold.processoutput import file_handlers


def test_npz_file(test_data):
    test_npz = Path(test_data.test_boltz_1_6BJ9_).joinpath(
        "predictions/test_mmseqs/pae_test_mmseqs_model_1.npz"
    )
    npz_file = file_handlers.NpzFile(test_npz)

    assert isinstance(npz_file.data, dict)
    assert "pae" in npz_file.data.keys()


def test_npy_file(test_data):
    test_npy = Path(test_data.test_chai1_6BJ9_).joinpath("pae_scores.npy")
    npy_file = file_handlers.NpyFile(test_npy)

    assert npy_file.data.shape == (2, 888, 888)


def test_cif_file(test_data):
    test_cif = Path(test_data.test_alphafold3_6BJ9_).joinpath(
        "seed-1_sample-0/model.cif"
    )
    cif_file = file_handlers.CifFile(test_cif)
    cif_file.check_clashes()

    assert cif_file.clashes == 0
    assert cif_file.average_plddt == pytest.approx(95.8, rel=1e-2)
    assert cif_file.h_score == 89

    plddt_regions_check = {
        "v_low": [(0, 1), (393, 394)],
        "low": [(2, 2), (395, 395)],
        "confident": [
            (3, 3),
            (134, 139),
            (209, 211),
            (396, 396),
            (527, 532),
            (601, 604),
        ],
        "v_high": [
            (4, 133),
            (140, 208),
            (212, 392),
            (397, 526),
            (533, 600),
            (605, 787),
        ],
    }
    plddts = cif_file.residue_plddts
    plddt_regions = get_plddt_regions(plddts)

    assert plddt_regions_check == plddt_regions
    assert cif_file.name == "model"
    assert cif_file.pathway == test_cif
    assert len(cif_file.plddts) > len(
        [
            plddts
            for plddts in cif_file.get_plddt_per_residue().values()
            for plddts in plddts
        ]
    )


def test_confidence_json_file(test_data):
    confidence_json = Path(test_data.test_alphafold3_6BJ9_).joinpath(
        "seed-1_sample-0/confidences.json"
    )
    confidence_file = file_handlers.ConfidenceJsonFile(confidence_json)

    assert "atom_chain_ids" in confidence_file.data
    assert "atom_plddts" in confidence_file.data
    assert "contact_probs" in confidence_file.data
    assert "pae" in confidence_file.data
