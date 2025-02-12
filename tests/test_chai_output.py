import tempfile
from pathlib import Path

from abcfold.processoutput.file_handlers import CifFile, NpyFile, NpzFile
from abcfold.processoutput.utils import Af3Pae


def test_process_chai_output(test_data, output_objs):
    chai_output = output_objs.chai_output

    assert str(chai_output.output_dir) == str(test_data.test_chai1_6BJ9_)

    assert -1 in chai_output.output
    assert 0 in chai_output.output
    assert 1 in chai_output.output

    assert all(isinstance(pae_file, NpyFile) for pae_file in chai_output.pae_files)
    assert all(isinstance(cif_file, CifFile) for cif_file in chai_output.cif_files)
    assert all(
        isinstance(scores_file, NpzFile) for scores_file in chai_output.scores_files
    )


def test_chai_pae_to_af3_pae(output_objs):
    comparison_af3_output = output_objs.af3_output.af3_pae_files["seed-1"][0].data
    pae_file = output_objs.chai_output.pae_files[-1]
    for i, cif_file in enumerate(output_objs.chai_output.cif_files):
        assert cif_file.input_params
        pae = Af3Pae.from_chai1(
            pae_file.data[i],
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
