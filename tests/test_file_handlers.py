import shutil
import tempfile
from pathlib import Path

import pytest

from abcfold.html.html_utils import get_plddt_regions
from abcfold.output import file_handlers


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
            (3, 4),
            (23, 23),
            (28, 28),
            (44, 44),
            (49, 49),
            (134, 141),
            (176, 176),
            (180, 180),
            (200, 200),
            (206, 206),
            (208, 214),
            (228, 228),
            (234, 234),
            (240, 240),
            (267, 267),
            (272, 272),
            (330, 330),
            (396, 397),
            (416, 416),
            (421, 421),
            (437, 437),
            (442, 442),
            (527, 534),
            (569, 569),
            (573, 573),
            (593, 593),
            (599, 599),
            (601, 607),
            (610, 610),
            (621, 621),
            (627, 627),
            (633, 633),
            (660, 660),
            (665, 665),
            (723, 723),
        ],
        "v_high": [
            (5, 22),
            (24, 27),
            (29, 43),
            (45, 48),
            (50, 133),
            (142, 175),
            (177, 179),
            (181, 199),
            (201, 205),
            (207, 207),
            (215, 227),
            (229, 233),
            (235, 239),
            (241, 266),
            (268, 271),
            (273, 329),
            (331, 392),
            (398, 415),
            (417, 420),
            (422, 436),
            (438, 441),
            (443, 526),
            (535, 568),
            (570, 572),
            (574, 592),
            (594, 598),
            (600, 600),
            (608, 609),
            (611, 620),
            (622, 626),
            (628, 632),
            (634, 659),
            (661, 664),
            (666, 722),
            (724, 787),
        ],
    }
    plddts = cif_file.residue_plddts
    plddt_regions = get_plddt_regions(plddts)

    print(f"PLDDT regions: {plddt_regions}")

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


def test_superpose_models(test_data):
    """
    Test the superpose_models function to ensure it correctly superposes two CifFiles.
    """

    with tempfile.TemporaryDirectory() as temp_dir:
        model_1 = Path(test_data.test_alphafold3_6BJ9_).joinpath(
            "seed-1_sample-0/model.cif"
        )
        model_2 = Path(test_data.test_boltz_1_6BJ9_).joinpath(
            "predictions/test_mmseqs/test_mmseqs_model_0.cif"
        )

        shutil.copyfile(model_1, f"{temp_dir}/model_1.cif")
        shutil.copyfile(model_2, f"{temp_dir}/model_2.cif")

        models = [Path(f"{temp_dir}/model_1.cif"), Path(f"{temp_dir}/model_2.cif")]
        file_handlers.superpose_models(models)

        superposed_atoms = []
        count = 0
        with open(f"{temp_dir}/model_2.cif", "r") as f:
            for line in f:
                if line.startswith("ATOM"):
                    fields = line.split()
                    if count <= 10:
                        superposed_atoms.append([fields[10], fields[11], fields[12]])
                    count += 1

        ref_superposed_atoms = [
            ["-4.804", "-5.203", "18.445"],
            ["-4.255", "-5.653", "19.697"],
            ["-3.629", "-7.029", "19.656"],
            ["-4.233", "-7.987", "19.137"],
            ["-2.464", "-7.105", "20.243"],
            ["-1.709", "-8.355", "20.246"],
            ["-1.191", "-8.799", "21.616"],
            ["-1.343", "-9.963", "21.992"],
            ["-0.512", "-8.272", "19.291"],
            ["0.309", "-7.153", "19.645"],
            ["-1.015", "-8.051", "17.846"],
        ]

        print(f"Superposed atoms: {superposed_atoms}")

        assert ref_superposed_atoms == superposed_atoms
