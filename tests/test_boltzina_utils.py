"""GPU-free tests for the boltzina affinity helpers."""

import tempfile
from pathlib import Path

import gemmi
import pytest
from Bio.PDB import MMCIFParser

from abcfold.affinity import boltzina_utils as bu

# A predicted complex with two protein chains (A, B) and an ACO ligand (C, D).
LIGAND_CIF = (
    "tests/test_data/boltz_6BJ9_seed-1/predictions/test_mmseqs/"
    "test_mmseqs_model_0.cif"
)


def _model():
    return MMCIFParser(QUIET=True).get_structure("m", LIGAND_CIF)[0]


def test_canonical_atom_names():
    names = bu.canonical_atom_names(["C", "C", "N", "Cl", "C", "cl"])
    assert names == ["C1", "C2", "N1", "CL1", "C3", "CL2"]
    # Names are short and unique
    assert len(set(names)) == len(names)


def test_smiles_heavy_atom_count():
    pytest.importorskip("rdkit")  # smiles_heavy_atom_count uses rdkit
    assert bu.smiles_heavy_atom_count("CCO") == 3          # ethanol
    assert bu.smiles_heavy_atom_count("O") == 1            # water
    assert bu.smiles_heavy_atom_count("not a smiles") is None


def test_call_with_supported_filters_and_raises():
    def f(a, b=2):
        return (a, b)

    # Unsupported kwargs (c) are dropped; supported ones passed through.
    assert bu._call_with_supported(f, a=1, b=3, c=99) == (1, 3)

    def g(x):
        return x

    # Required positional not covered by any candidate -> TypeError
    with pytest.raises(TypeError):
        bu._call_with_supported(g, y=1)


def test_resolve_cache_path_explicit():
    p = Path("/tmp/some/boltz/cache")
    assert bu.resolve_cache_path(p) == p


def test_detect_ligands_finds_aco():
    ligs = bu.detect_ligands(_model(), include_additives=True)
    resnames = {c["resname"] for c in ligs}
    assert "ACO" in resnames
    aco = next(c for c in ligs if c["resname"] == "ACO")
    assert aco["chain_id"] in ("C", "D")
    assert aco["num_heavy_atoms"] == 51


def test_extract_protein_sequences():
    seqs = bu.extract_protein_sequences(_model())
    assert set(seqs) == {"A", "B"}
    assert all(len(s) > 50 for s in seqs.values())
    assert set("".join(seqs.values())) <= set("ACDEFGHIKLMNPQRSTVWYX")


def test_select_ligand_by_chain_and_errors():
    model = _model()
    chosen = bu.select_ligand(model, ligand_chain="C")
    assert chosen["resname"] == "ACO" and chosen["chain_id"] == "C"

    # No ligand in a nonexistent chain -> ValueError
    with pytest.raises(ValueError):
        bu.select_ligand(model, ligand_chain="Z")

    # No ligand at all -> ValueError (drop the ligand chains first)
    for cid in ("C", "D"):
        if cid in [c.id for c in model]:
            model.detach_child(cid)
    with pytest.raises(ValueError):
        bu.select_ligand(model)


def test_write_ligand_pdb():
    model = _model()
    ligand = bu.select_ligand(model, ligand_chain="C")
    with tempfile.TemporaryDirectory() as td:
        out = bu.write_ligand_pdb(model, ligand, Path(td) / "lig.pdb")
        assert out.exists()
        lines = out.read_text().splitlines()
        atom_lines = [ln for ln in lines if ln.startswith(("ATOM", "HETATM"))]
        assert len(atom_lines) == 51  # heavy atoms only

        # A ligand that isn't in the structure -> ValueError
        bogus = dict(ligand, chain_id="C", resseq=999999)
        with pytest.raises(ValueError):
            bu.write_ligand_pdb(model, bogus, Path(td) / "bad.pdb")


def test_relabel_ligand_in_cif():
    cif_in = (
        "data_test\n"
        "loop_\n"
        "_atom_site.group_PDB\n"
        "_atom_site.label_atom_id\n"
        "_atom_site.label_comp_id\n"
        "_atom_site.label_asym_id\n"
        "_atom_site.label_seq_id\n"
        "_atom_site.Cartn_x\n"
        "_atom_site.Cartn_y\n"
        "_atom_site.Cartn_z\n"
        "HETATM C1 LIG0 A 1 0.0 0.0 0.0\n"
        "HETATM C2 LIG0 A 1 1.0 0.0 0.0\n"
    )
    with tempfile.TemporaryDirectory() as td:
        src = Path(td) / "in.cif"
        dst = Path(td) / "out.cif"
        src.write_text(cif_in)
        bu.relabel_ligand_in_cif(src, "LIG0", "LIG", dst)
        text = dst.read_text()
        assert "LIG0" not in text
        assert " LIG " in text


def test_normalize_complex_cif_adds_entities():
    with tempfile.TemporaryDirectory() as td:
        out = Path(td) / "norm.cif"
        bu.normalize_complex_cif(Path(LIGAND_CIF), out)
        assert out.exists()
        # gemmi can parse it and it carries entity records
        st = gemmi.read_structure(str(out))
        assert len(st.entities) > 0
