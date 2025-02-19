import tempfile
from pathlib import Path

from abcfold.plots.pae_plot import create_pae_plots
from abcfold.plots.plddt_plot import plot_plddt
from abcfold.plots.plotter import get_model_sequence_data


def test_plddt_plot(output_objs):

    af3_files = output_objs.af3_output.cif_files["seed-1"]
    boltz_files = output_objs.boltz_output.cif_files
    chai_files = output_objs.chai_output.cif_files

    assert len(af3_files) == len(boltz_files) == len(chai_files)
    plot_files = {
        "Alphafold3": af3_files,
        "Boltz-1": boltz_files,
        "Chai-1": chai_files,
    }

    with tempfile.TemporaryDirectory() as temp_dir:
        plot_plddt(
            plot_files,
            output_name=f"{temp_dir}/test.html",
        )

        assert Path(f"{temp_dir}/test.html").exists()


def test_pae_plots(output_objs):
    outputs = [
        output_objs.af3_output,
        output_objs.boltz_output,
        output_objs.chai_output,
    ]

    with tempfile.TemporaryDirectory() as temp_dir_str:
        temp_dir = Path(temp_dir_str)
        plot_pathways = create_pae_plots(outputs, output_dir=temp_dir)

        assert len(plot_pathways) == 6
        values = [Path(value).name for value in plot_pathways.values()]

        assert "confidences_seed-1_sample-0_af3_pae_plot.html" in values
        assert "confidences_seed-1_sample-1_af3_pae_plot.html" in values
        assert "test_mmseqs_model_0_af3_pae_pae_plot.html" in values
        assert "test_mmseqs_model_1_af3_pae_pae_plot.html" in values
        assert "pred.model_idx_0_af3_pae_pae_plot.html" in values
        assert "pred.model_idx_1_af3_pae_pae_plot.html" in values

        assert (
            "tests/test_data/alphafold3_6BJ9/seed-1_sample-0/\
model.cif"
            in plot_pathways
        )
        assert (
            "tests/test_data/alphafold3_6BJ9/seed-1_sample-1/\
model.cif"
            in plot_pathways
        )
        assert (
            "tests/test_data/boltz-1_6BJ9/predictions/test_mmseqs/\
test_mmseqs_model_0.cif"
            in plot_pathways
        )
        assert (
            "tests/test_data/boltz-1_6BJ9/predictions/test_mmseqs/\
test_mmseqs_model_1.cif"
            in plot_pathways
        )
        assert "tests/test_data/chai1_6BJ9/pred.model_idx_0.cif" in plot_pathways
        assert "tests/test_data/chai1_6BJ9/pred.model_idx_1.cif" in plot_pathways

        assert len(list(temp_dir.glob("*.html"))) == 6


def test_get_sequence_data(output_objs):
    af3_files = output_objs.af3_output.cif_files["seed-1"]
    boltz_files = output_objs.boltz_output.cif_files
    chai_files = output_objs.chai_output.cif_files

    cif_files = []

    [cif_files.extend(files) for files in [af3_files, boltz_files, chai_files]]

    outputdic = get_model_sequence_data(cif_files)

    assert outputdic == {
        "A": "GTGSRPITDVVFVGAARTPIGSFRSAFNNVPVTVLGREALKGALKNANVKPSLVQEAFIGVVVPSNAGQGPA\
RQVVLGAGCDVSTVVTAVNKMCASGMKAIACAASILQLDLQEMVVAGGMESMSCVPFYLPRGEIPFGGTKLIDGIPRDGLNDVYND\
ILMGACADKVAKQFAITREEQDKYAILSYKRSAAAWKEGIFAKEIIPLEVTQGKKTITVEEDEEYKKVNFEKIPKLKPAFTSEGSV\
TAANASTLNDGAAMVVMTTVDGAKKHGLKPLARMLAYGDAATHPIDFGIAPASVIPKVLKLAGLQIKDIDLWEINEAFAVVPLYTM\
KTLGLDESKVNIHGGAVSLGHPIGMSGARIVGHLVHTLKPGQKGCAAICNGGGGAGGMIIEKL",
        "B": "GTGSRPITDVVFVGAARTPIGSFRSAFNNVPVTVLGREALKGALKNANVKPSLVQEAFIGVVVPSNAGQGPA\
RQVVLGAGCDVSTVVTAVNKMCASGMKAIACAASILQLDLQEMVVAGGMESMSCVPFYLPRGEIPFGGTKLIDGIPRDGLNDVYND\
ILMGACADKVAKQFAITREEQDKYAILSYKRSAAAWKEGIFAKEIIPLEVTQGKKTITVEEDEEYKKVNFEKIPKLKPAFTSEGSV\
TAANASTLNDGAAMVVMTTVDGAKKHGLKPLARMLAYGDAATHPIDFGIAPASVIPKVLKLAGLQIKDIDLWEINEAFAVVPLYTM\
KTLGLDESKVNIHGGAVSLGHPIGMSGARIVGHLVHTLKPGQKGCAAICNGGGGAGGMIIEKL",
        "C": "NCNCCCNNCNCCOCOPOOOCOCOPOOOPOOOCCCCCOCONCCCONCCSCOC",
        "D": "NCNCCCNNCNCCOCOPOOOCOCOPOOOPOOOCCCCCOCONCCCONCCSCOC",
    }
