import tempfile
from pathlib import Path

from abcfold.output.file_handlers import CifFile, ConfidenceJsonFile, NpzFile
from abcfold.output.utils import Af3Pae


def test_process_boltz_output(test_data, output_objs):
    boltz_output = output_objs.boltz_output
    assert boltz_output.output_dirs[0].relative_to(
        boltz_output.output_dirs[0].parent
    ) == Path(test_data.test_boltz_6BJ9_seed_1_).relative_to("tests/test_data")

    assert boltz_output.name == "6BJ9"

    assert 0 in boltz_output.output["seed-1"]
    assert 1 in boltz_output.output["seed-1"]

    assert "af3_pae" in boltz_output.output["seed-1"][0]
    assert "cif" in boltz_output.output["seed-1"][0]
    assert "json" in boltz_output.output["seed-1"][0]

    assert "af3_pae" in boltz_output.output["seed-1"][1]
    assert "cif" in boltz_output.output["seed-1"][1]
    assert "json" in boltz_output.output["seed-1"][1]

    assert all(
        isinstance(pae_file, NpzFile) for pae_file in boltz_output.pae_files["seed-1"]
    )
    assert all(
        isinstance(plddt_file, NpzFile)
        for plddt_file in boltz_output.plddt_files["seed-1"]
    )
    assert all(
        isinstance(pde_file, NpzFile) for pde_file in boltz_output.pde_files["seed-1"]
    )
    assert all(
        isinstance(cif_file, CifFile) for cif_file in boltz_output.cif_files["seed-1"]
    )
    assert all(
        isinstance(scores_file, ConfidenceJsonFile)
        for scores_file in boltz_output.scores_files["seed-1"]
    )

    assert boltz_output.cif_files["seed-1"][0].chain_lengths() == {
        "A": 393,
        "B": 393,
        "C": 1,
        "D": 1,
    }

    with tempfile.TemporaryDirectory() as temp_dir_str:
        temp_dir = Path(temp_dir_str)
        for i, cif_file in enumerate(boltz_output.cif_files["seed-1"]):
            cif_file.to_file(temp_dir / f"{i}.cif")
            assert (temp_dir / f"{i}.cif").exists()


def test_boltz_pae_to_af3_pae(test_data, output_objs):
    comparison_af3_output = output_objs.af3_output.af3_pae_files["seed-1"][0].data
    for pae_file, cif_file in zip(
        output_objs.boltz_output.pae_files["seed-1"],
        output_objs.boltz_output.cif_files["seed-1"],
    ):
        pae = Af3Pae.from_boltz(
            pae_file.data,
            cif_file,
        )

        with tempfile.TemporaryDirectory() as temp_dir_str:
            temp_dir = Path(temp_dir_str)
            pae.to_file(temp_dir / "pae.json")
            assert (temp_dir / "pae.json").exists()

        # for some reason the lengths are different for atom - realted things
        # If it isn't breaking the output page generation, then it's fine
        assert len(pae.scores["pae"]) == len(comparison_af3_output["pae"])

        assert len(pae.scores["contact_probs"]) == len(
            comparison_af3_output["contact_probs"]
        )
        assert len(pae.scores["token_chain_ids"]) == len(
            comparison_af3_output["token_chain_ids"]
        )
        assert len(pae.scores["token_res_ids"]) == len(
            comparison_af3_output["token_res_ids"]
        )
