import tempfile
from pathlib import Path

from abcfold.html.html_utils import get_model_sequence_data
from abcfold.plots.pae_plot import create_pae_plots
from abcfold.plots.plddt_plot import plot_plddt


def test_plddt_plot(output_objs):

    af3_files = output_objs.af3_output.cif_files["seed-1"]
    boltz_files = output_objs.boltz_output.cif_files["seed-1"]
    chai_files = output_objs.chai_output.cif_files["seed-1"]
    protenix_files = output_objs.protenix_output.cif_files["seed-1"]
    openfold_files = output_objs.openfold_output.cif_files["seed-1"]

    assert all(len(files) == len(af3_files) for files in [
        boltz_files,
        chai_files,
        protenix_files,
        openfold_files
    ])

    plot_files = {
        "Alphafold3": af3_files,
        "Boltz": boltz_files,
        "Chai-1": chai_files,
        "Protenix": protenix_files,
        "OpenFold3": openfold_files
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
        output_objs.protenix_output,
        output_objs.openfold_output
    ]

    with tempfile.TemporaryDirectory() as temp_dir_str:
        temp_dir = Path(temp_dir_str)
        plot_pathways = create_pae_plots(outputs, output_dir=temp_dir)

        assert len(plot_pathways) == 10
        values = [Path(value).name for value in plot_pathways.values()]

        assert "confidences_seed-1_sample-0_af3_pae_plot.html" in values
        assert "confidences_seed-1_sample-1_af3_pae_plot.html" in values
        assert "pae_test_mmseqs_model_0_test_mmseqs_af3_pae_plot.html" in values
        assert "pae_test_mmseqs_model_1_test_mmseqs_af3_pae_plot.html" in values
        assert "pae_scores_model_0_chai1_6BJ9_seed-1_af3_pae_plot.html" in values
        assert "pae_scores_model_1_chai1_6BJ9_seed-1_af3_pae_plot.html" in values
        assert "6BJ9_full_data_sample_0_predictions_af3_pae_plot.html" in values
        assert "6BJ9_full_data_sample_1_predictions_af3_pae_plot.html" in values
        assert "6BJ9_seed_1_sample_1_confidences_seed_1_af3_pae_plot.html" in values
        assert "6BJ9_seed_1_sample_2_confidences_seed_1_af3_pae_plot.html" in values

        assert (
            any(
                "alphafold3_6BJ9/seed-1_sample-0/model.cif" in x for x in plot_pathways
            )
        )
        assert (
            any(
                "alphafold3_6BJ9/seed-1_sample-1/model.cif" in x for x in plot_pathways
            )
        )
        assert (
            any(
                "boltz_6BJ9_seed-1/predictions/test_mmseqs/test_mmseqs_model_0.cif"
                in x for x in plot_pathways
            )
        )
        assert (
            any(
                "boltz_6BJ9_seed-1/predictions/test_mmseqs/test_mmseqs_model_1.cif"
                in x for x in plot_pathways
            )
        )
        assert (
            any("chai1_6BJ9_seed-1/pred.model_idx_0.cif" in x for x in plot_pathways)
        )
        assert (
            any("chai1_6BJ9_seed-1/pred.model_idx_1.cif" in x for x in plot_pathways)
        )
        assert (
            any("protenix_6BJ9_seed-1/6BJ9/seed_1/predictions/6BJ9_sample_0.cif"
                in x for x in plot_pathways)
        )
        assert (
            any("protenix_6BJ9_seed-1/6BJ9/seed_1/predictions/6BJ9_sample_1.cif"
                in x for x in plot_pathways)
        )
        assert (
            any("openfold_6BJ9_seed-1/6BJ9/seed_1/6BJ9_seed_1_sample_1_model"
                in x for x in plot_pathways)
        )
        assert (
            any("openfold_6BJ9_seed-1/6BJ9/seed_1/6BJ9_seed_1_sample_2_model"
                in x for x in plot_pathways)
        )

        assert len(list(temp_dir.glob("*.html"))) == 10


def test_get_sequence_data(output_objs):
    af3_files = output_objs.af3_output.cif_files["seed-1"]
    boltz_files = output_objs.boltz_output.cif_files["seed-1"]
    chai_files = output_objs.chai_output.cif_files["seed-1"]
    protenix_files = output_objs.protenix_output.cif_files["seed-1"]
    openfold_files = output_objs.openfold_output.cif_files["seed-1"]

    cif_files = [item for sublist in [af3_files,
                                      boltz_files,
                                      chai_files,
                                      openfold_files,
                                      protenix_files]
                 for item in sublist]

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
