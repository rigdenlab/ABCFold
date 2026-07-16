"""GPU-free tests for the batch affinity helper (no Boltz env needed)."""

import tempfile
from pathlib import Path

from abcfold.affinity.run_affinity import (extract_affinity_msa,
                                           run_boltz_affinity)


def test_extract_affinity_msa_from_path():
    params = {
        "sequences": [
            {"protein": {"id": "A", "sequence": "MKV",
                         "unpairedMsaPath": "/some/msa.a3m"}},
        ]
    }
    with tempfile.TemporaryDirectory() as td:
        assert extract_affinity_msa(params, td) == "/some/msa.a3m"


def test_extract_affinity_msa_inlines_and_writes():
    a3m = ">q\nMKV\n>h1\nMKA\n"
    params = {"sequences": [{"protein": {"id": "A", "sequence": "MKV",
                                         "unpairedMsa": a3m}}]}
    with tempfile.TemporaryDirectory() as td:
        out = extract_affinity_msa(params, td)
        assert out is not None
        p = Path(out)
        assert p.exists() and p.name == "reused_msa.a3m"
        assert p.read_text() == a3m


def test_extract_affinity_msa_none_for_multichain():
    params = {
        "sequences": [
            {"protein": {"id": "A", "sequence": "MKV", "unpairedMsa": ">q\nM\n"}},
            {"protein": {"id": "B", "sequence": "GGG", "unpairedMsa": ">q\nG\n"}},
        ]
    }
    with tempfile.TemporaryDirectory() as td:
        # A single alignment can't be shared across a heteromer -> None
        assert extract_affinity_msa(params, td) is None


def test_extract_affinity_msa_none_without_protein_or_msa():
    with tempfile.TemporaryDirectory() as td:
        assert extract_affinity_msa({"sequences": []}, td) is None
        no_msa = {"sequences": [{"protein": {"id": "A", "sequence": "M"}}]}
        assert extract_affinity_msa(no_msa, td) is None


def test_run_boltz_affinity_no_models_or_ligand():
    with tempfile.TemporaryDirectory() as td:
        # No models -> empty result, no subprocess launched
        assert run_boltz_affinity([], td, smiles="CCO") == {}
        # No ligand chemistry -> skipped, empty result
        model = Path(td) / "m.cif"
        model.write_text("x")
        assert run_boltz_affinity([model], td) == {}


def test_run_boltz_affinity_parses_results_csv(monkeypatch):
    import abcfold.affinity.run_affinity as ra

    with tempfile.TemporaryDirectory() as td_str:
        td = Path(td_str)
        model = td / "model_0.cif"
        model.write_text("x")

        # Pre-create the CSV the scorer subprocess would have written.
        affinity_dir = td / "affinity"
        affinity_dir.mkdir()
        (affinity_dir / "boltz_affinity_results.csv").write_text(
            "input_model,affinity_pred_value,affinity_probability_binary\n"
            f"{model.resolve()},1.5,0.8\n"
        )

        # Stub the subprocess so no Boltz env / GPU is needed.
        class _Result:
            returncode = 0
            stdout = ""
            stderr = ""

        monkeypatch.setattr(ra.subprocess, "run", lambda *a, **k: _Result())

        scores = run_boltz_affinity([model], td, smiles="CCO")
        key = str(model.resolve())
        assert scores[key]["affinity_pred_value"] == 1.5
        assert scores[key]["affinity_probability_binary"] == 0.8
