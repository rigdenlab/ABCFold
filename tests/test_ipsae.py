import pytest

from abcfold.scripts.ipsae import Ipsae


def test_af3_ipsae(output_objs):
    af3_file = output_objs.af3_output.cif_files["seed-1"][0]
    af3_pae_file = output_objs.af3_output.af3_pae_files["seed-1"][0]

    ipsae_calculator = Ipsae(
        af3_file,
        af3_pae_file,
        pae_cutoff=10.0,
        distance_cutoff=5.0
    )
    scores = ipsae_calculator.main(verbose=False, output_csv=None)

    assert "pdockq" in scores
    assert "pdockq2" in scores
    assert "ligand_interface_scores" in scores
    assert "iptm_ipsae_scores" in scores

    assert scores["pdockq"]["A"]["B"] == pytest.approx(0.7226, rel=1e-3)
    assert scores["pdockq2"]["A"]["B"] == pytest.approx(0.733, rel=1e-3)
    assert scores["ligand_interface_scores"]["A"]["B"] == pytest.approx(
        0.5385, rel=1e-3
    )

    iptm = scores["iptm_ipsae_scores"]
    assert iptm["iptm_d0chn_asym"]["A"]["B"] == pytest.approx(
        0.8444376339, rel=1e-6
    )
    assert iptm["ipsae_d0res_asym"]["A"]["B"] == pytest.approx(
        0.7545828078, rel=1e-6
    )

    assert iptm["unique_residues_chain1"]["A"]["B"] == 391
    assert iptm["unique_residues_chain2"]["A"]["B"] == 393


def test_boltz_ipsae(output_objs):
    boltz_file = output_objs.boltz_output.cif_files["seed-1"][0]
    boltz_pae_file = output_objs.boltz_output.af3_pae_files["seed-1"][0]

    ipsae_calculator = Ipsae(
        boltz_file,
        boltz_pae_file,
        pae_cutoff=10.0,
        distance_cutoff=5.0
    )
    scores = ipsae_calculator.main(verbose=False, output_csv=None)

    assert "pdockq" in scores
    assert "pdockq2" in scores
    assert "ligand_interface_scores" in scores
    assert "iptm_ipsae_scores" in scores

    assert scores["pdockq"]["A"]["B"] == pytest.approx(0.7248, rel=1e-3)
    assert scores["pdockq2"]["A"]["B"] == pytest.approx(0.5144, rel=1e-3)
    assert scores["ligand_interface_scores"]["A"]["B"] == pytest.approx(
        0.7273, rel=1e-3
    )

    iptm = scores["iptm_ipsae_scores"]
    assert iptm["iptm_d0chn_asym"]["A"]["B"] == pytest.approx(
        0.9633755758902331, rel=1e-6
    )
    assert iptm["ipsae_d0res_asym"]["A"]["B"] == pytest.approx(
        0.945514520263416, rel=1e-6
    )

    assert iptm["unique_residues_chain1"]["A"]["B"] == 389
    assert iptm["unique_residues_chain2"]["A"]["B"] == 390


def test_chai_ipsae(output_objs):
    chai_file = output_objs.chai_output.cif_files["seed-1"][0]
    chai_pae_file = output_objs.chai_output.af3_pae_files["seed-1"][0]

    ipsae_calculator = Ipsae(
        chai_file,
        chai_pae_file,
        pae_cutoff=10.0,
        distance_cutoff=5.0
    )
    scores = ipsae_calculator.main(verbose=False, output_csv=None)

    assert "pdockq" in scores
    assert "pdockq2" in scores
    assert "ligand_interface_scores" in scores
    assert "iptm_ipsae_scores" in scores

    assert scores["pdockq"]["A"]["B"] == pytest.approx(0.721, rel=1e-3)
    assert scores["pdockq2"]["A"]["B"] == pytest.approx(0.8039, rel=1e-3)
    assert scores["ligand_interface_scores"]["A"]["B"] == pytest.approx(
        0.7401, rel=1e-3
    )

    iptm = scores["iptm_ipsae_scores"]
    assert iptm["iptm_d0chn_asym"]["A"]["B"] == pytest.approx(
        0.9797472978285606, rel=1e-6
    )
    assert iptm["ipsae_d0res_asym"]["A"]["B"] == pytest.approx(
        0.9679382445215664, rel=1e-6
    )

    assert iptm["unique_residues_chain1"]["A"]["B"] == 391
    assert iptm["unique_residues_chain2"]["A"]["B"] == 393


def test_protenix_ipsae(output_objs):
    protenix_file = output_objs.protenix_output.cif_files["seed-1"][0]
    protenix_pae_file = output_objs.protenix_output.af3_pae_files["seed-1"][0]

    ipsae_calculator = Ipsae(
        protenix_file,
        protenix_pae_file,
        pae_cutoff=10.0,
        distance_cutoff=5.0
    )
    scores = ipsae_calculator.main(verbose=False, output_csv=None)

    assert "pdockq" in scores
    assert "pdockq2" in scores
    assert "ligand_interface_scores" in scores
    assert "iptm_ipsae_scores" in scores

    assert scores["pdockq"]["A"]["B"] == pytest.approx(0.7202, rel=1e-3)
    assert scores["pdockq2"]["A"]["B"] == pytest.approx(0.7041, rel=1e-3)
    assert scores["ligand_interface_scores"]["A"]["B"] == pytest.approx(
        0.6987, rel=1e-3
    )

    iptm = scores["iptm_ipsae_scores"]
    assert iptm["iptm_d0chn_asym"]["A"]["B"] == pytest.approx(
        0.9689027305100429, rel=1e-6
    )
    assert iptm["ipsae_d0res_asym"]["A"]["B"] == pytest.approx(
        0.9517480877376054, rel=1e-6
    )

    assert iptm["unique_residues_chain1"]["A"]["B"] == 391
    assert iptm["unique_residues_chain2"]["A"]["B"] == 391
