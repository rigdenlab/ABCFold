import gzip
import json
import tempfile
from pathlib import Path

import gemmi

from abcfold.chai1.chai_templates import prepare_chai_templates
from abcfold.scripts.abc_script_utils import get_mmcif


def _make_template_mmcif(tmpdir):
    """Build a valid single-chain template mmCIF from the 6BJ9 test data."""
    cif = "tests/test_data/6BJ9.cif"
    chain = gemmi.read_structure(cif)[0][0].name
    return get_mmcif(cif, "6BJ9", chain, 1, 200, tmpdir=tmpdir)


def test_prepare_chai_custom_templates():
    with tempfile.TemporaryDirectory() as td:
        mmcif = _make_template_mmcif(td)
        input_params = {
            "sequences": [
                {
                    "protein": {
                        "id": ["A", "B"],
                        "sequence": "GMRES",
                        "templates": [
                            {
                                "mmcif": mmcif,
                                "queryIndices": [0, 1, 2, 3, 4],
                                "templateIndices": [10, 11, 12, 13, 14],
                            }
                        ],
                    }
                },
                {"protein": {"id": "C", "sequence": "YANEN"}},  # no templates
            ]
        }

        work_dir = Path(td) / "work"
        work_dir.mkdir()
        m8, store = prepare_chai_templates(
            input_params, work_dir=work_dir, existing_m8=None
        )

        # No mmseqs m8 in, none out; the m8 is assembled later in the Chai env
        assert m8 is None

        # A custom template CIF was written and gzipped into the store
        assert store is not None
        cif_gz = store / "CT00.cif.gz"
        assert cif_gz.exists()

        # It is a valid, gemmi-parseable CIF with entities (the Boltz/Chai fix)
        with gzip.open(cif_gz, "rt") as fh:
            st = gemmi.make_structure_from_block(gemmi.cif.read_string(fh.read())[0])
        assert len(st.entities) > 0

        # A manifest records which query chains the template maps to (A, B; not C)
        manifest = json.loads((store / "custom_templates.json").read_text())
        assert "CT00" in manifest
        assert manifest["CT00"]["query_chains"] == ["A", "B"]
        assert manifest["CT00"]["template_chain"]


def test_prepare_chai_prepopulates_from_local_db():
    with tempfile.TemporaryDirectory() as td_str:
        td = Path(td_str)
        # Fake existing mmseqs m8 with a single PDB hit (6bj9 chain A)
        existing_m8 = td / "all_chains.m8"
        existing_m8.write_text(
            "A\t6bj9_A\t100.0\t50\t0\t0\t1\t50\t1\t50\t1e-9\t100.0\thit\n"
        )

        # Fake local mmseqs database layout: pdb/divided/bj/6bj9.cif.gz
        db = td / "db"
        cif_dir = db / "pdb" / "divided" / "bj"
        cif_dir.mkdir(parents=True)
        with gzip.open(cif_dir / "6bj9.cif.gz", "wt") as fh:
            fh.write(Path("tests/test_data/6BJ9.cif").read_text())

        work_dir = td / "work"
        work_dir.mkdir()
        combined_m8, store = prepare_chai_templates(
            {"sequences": []},
            work_dir=work_dir,
            existing_m8=existing_m8,
            mmseqs_database=db,
        )

        # The PDB CIF was pre-populated under the uppercased name Chai expects
        assert store is not None
        assert (store / "6BJ9.cif.gz").exists()


def test_prepare_chai_no_templates_is_noop():
    with tempfile.TemporaryDirectory() as td:
        work_dir = Path(td) / "work"
        work_dir.mkdir()
        combined_m8, store = prepare_chai_templates(
            {"sequences": [{"protein": {"id": "A", "sequence": "GMRES"}}]},
            work_dir=work_dir,
            existing_m8=None,
        )
        # Nothing to add: no combined m8, no store to serve
        assert combined_m8 is None
        assert store is None
