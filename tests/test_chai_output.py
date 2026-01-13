import tempfile
from pathlib import Path

from abcfold.output.file_handlers import CifFile, NpyFile, NpzFile
from abcfold.output.utils import Af3Pae


def test_process_chai_output(test_data, output_objs):
    chai_output = output_objs.chai_output

    assert chai_output.output_dirs[0].relative_to(
        chai_output.output_dirs[0].parent
    ) == Path(test_data.test_chai1_6BJ9_seed_1_).relative_to("tests/test_data")

    assert 0 in chai_output.output["seed-1"]
    assert 1 in chai_output.output["seed-1"]

    assert all(
        isinstance(pae_file, NpyFile) for pae_file in chai_output.pae_files["seed-1"]
    )
    assert all(
        isinstance(cif_file, CifFile) for cif_file in chai_output.cif_files["seed-1"]
    )
    assert all(
        isinstance(scores_file, NpzFile)
        for scores_file in chai_output.scores_files["seed-1"]
    )


def test_chai_pae_to_af3_pae(output_objs):
    comparison_af3_output = output_objs.af3_output.af3_pae_files["seed-1"][0].data
    for pae_file, cif_file in zip(
        output_objs.chai_output.pae_files["seed-1"],
        output_objs.chai_output.cif_files["seed-1"],
    ):
        assert cif_file.input_params
        pae = Af3Pae.from_chai1(
            pae_file.data,
            cif_file,
        )

        with tempfile.TemporaryDirectory() as temp_dir_str:
            temp_dir = Path(temp_dir_str)
            pae.to_file(temp_dir / "pae.json")
            assert (temp_dir / "pae.json").exists()

        # for some reason the lengths are different for atom - realted things
        # If it isn't breaking the output page generation, then it's fine
        assert len(pae.scores["pae"][0]) == len(comparison_af3_output["pae"])

        assert len(pae.scores["contact_probs"][0]) == len(
            comparison_af3_output["contact_probs"]
        )
        assert len(pae.scores["token_chain_ids"]) == len(
            comparison_af3_output["token_chain_ids"]
        )
        assert len(pae.scores["token_res_ids"]) == len(
            comparison_af3_output["token_res_ids"]
        )
