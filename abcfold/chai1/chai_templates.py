"""Template handling helpers for Chai-1.

Chai-1 only accepts templates through an m8 "hits" file: for every hit it splits
``subject_id`` into ``PDBID_chain`` and calls ``rcsb.download_cif_file(PDBID)``,
which fetches ``{PDBID}.cif.gz`` from RCSB into a per-run cache folder using
``download_if_not_exists`` (i.e. it skips the download when the file is already
present).

We exploit that skip to do two things without ever changing chai-lab:

1. **Pre-populate** the PDB template CIFs ABCFold already has (from a local
   mmseqs template database) so Chai doesn't re-download them from RCSB.
2. **Inject custom templates** (``--custom_template`` / non-PDB structures):
   we give each one a synthetic id, drop ``{ID}.cif.gz`` in the store, and
   fabricate an m8 row pointing at it. Chai then loads it as a template with no
   RCSB call (which would 404 for a non-PDB id).

Both funnel through a single *store* directory of ``{ID}.cif.gz`` files. The
Chai wrapper (``chai.py``) monkeypatches ``download_cif_file`` to serve files
from this store before falling back to the real download.

Custom-template CIFs are normalised with gemmi's ``setup_entities()`` (and a
``full_sequence`` backfill) so chai-lab's gemmi-based reader can parse the
polymer chain -- the same fix used for Boltz/affinity template CIFs.
"""

import gzip
import logging
from pathlib import Path
from typing import List, Optional, Tuple, Union

logger = logging.getLogger("logger")

# m8 columns, in order, as parsed by chai-lab's parse_m8_file.
_M8_COLUMNS = 13


def _normalise_and_gzip_cif(mmcif: str, out_gz: Path) -> Optional[str]:
    """Normalise an inline mmCIF and write it gzipped to ``out_gz``.

    Runs gemmi ``setup_entities()`` and backfills each polymer entity's
    ``full_sequence`` so chai-lab's ``gemmi.read_structure`` +
    ``chain.get_polymer()`` can resolve the polymer. Returns the auth name of
    the first polymer chain (what Chai indexes via ``structure[0][chain]``), or
    ``None`` if the CIF couldn't be parsed.
    """
    try:
        import gemmi

        doc = gemmi.cif.read_string(mmcif)
        st = gemmi.make_structure_from_block(doc[0])
        if len(st) == 0:
            raise ValueError("template mmCIF has no models")
        st.setup_entities()

        model0 = st[0]
        for entity in st.entities:
            if entity.entity_type.name != "Polymer" or entity.full_sequence:
                continue
            if not entity.subchains:
                continue
            first_subchain = entity.subchains[0]
            seq = [
                residue.name
                for chain in model0
                for residue in chain
                if residue.subchain == first_subchain
            ]
            if seq:
                entity.full_sequence = seq

        # Chai indexes the template by auth chain name (structure[0][chain]).
        chain_name = model0[0].name if len(model0) else None

        out_gz.parent.mkdir(parents=True, exist_ok=True)
        cif_text = st.make_mmcif_document().as_string()
        with gzip.open(out_gz, "wt") as fh:
            fh.write(cif_text)
        return chain_name
    except Exception as exc:  # noqa: BLE001
        logger.warning("Could not normalise custom template CIF: %s", exc)
        return None


def _copy_local_pdb_cifs(
    existing_m8: Path, store_dir: Path, mmseqs_database: Path
) -> int:
    """Pre-populate the store with PDB CIFs from a local mmseqs database.

    For every hit in ``existing_m8`` (subject_id ``PDBID_chain``) copy the
    matching ``{pdbid}.cif.gz`` from ``mmseqs_database/pdb/divided`` into the
    store as ``{PDBID}.cif.gz`` (the name Chai's download_cif_file expects), so
    Chai skips the RCSB download. Returns how many were pre-populated.
    """
    import shutil

    count = 0
    seen = set()
    for line in Path(existing_m8).read_text().splitlines():
        if not line.strip():
            continue
        cols = line.split("\t")
        if len(cols) < 2:
            continue
        subject = cols[1]
        pdb_id = subject.split("_")[0].lower()
        if pdb_id in seen or len(pdb_id) != 4:
            continue
        seen.add(pdb_id)
        src = Path(mmseqs_database).joinpath(
            "pdb", "divided", pdb_id[1:3], f"{pdb_id}.cif.gz"
        )
        if not src.exists():
            continue
        dest = store_dir / f"{pdb_id.upper()}.cif.gz"
        if not dest.exists():
            shutil.copyfile(src, dest)
            count += 1
    return count


def _fabricate_m8_row(
    query_id: str,
    subject_id: str,
    query_indices: List[int],
    template_indices: List[int],
) -> Optional[str]:
    """Build a single tab-separated m8 row for a custom template hit.

    Uses the query/template residue index mapping ABCFold already computed when
    adding the template (``align_and_map``) to fill the alignment spans. Chai
    re-aligns with kalign, so exact identity/e-value are not critical, but the
    spans must be sane (Chai slices the template sequence with subject_start/end
    and checks the query span). 1-indexed to match m8 convention (Chai converts
    the *_start columns back to 0-indexed).
    """
    if not query_indices or not template_indices:
        return None
    q_start, q_end = min(query_indices) + 1, max(query_indices) + 1
    s_start, s_end = min(template_indices) + 1, max(template_indices) + 1
    length = max(q_end - q_start + 1, 1)
    cols = [
        query_id,        # query_id  (must match Chai chain entity_name)
        subject_id,      # subject_id = {ID}_{chain}
        "100.0",         # pident
        str(length),     # length
        "0",             # mismatch
        "0",             # gapopen
        str(q_start),    # query_start (1-indexed)
        str(q_end),      # query_end
        str(s_start),    # subject_start (1-indexed)
        str(s_end),      # subject_end
        "1e-9",          # evalue
        "100.0",         # bitscore
        "custom",        # comment
    ]
    assert len(cols) == _M8_COLUMNS
    return "\t".join(cols)


def prepare_chai_templates(
    input_params: dict,
    work_dir: Path,
    existing_m8: Optional[Union[str, Path]] = None,
    mmseqs_database: Optional[Union[str, Path]] = None,
) -> Tuple[Optional[Path], Optional[Path]]:
    """Build a Chai template store + combined m8 for pre-pop and custom templates.

    Args:
        input_params: The AF3-style input dict (its protein entries may carry
            inline ``templates`` with mmCIF payloads).
        work_dir: Directory to write the store and combined m8 into.
        existing_m8: The mmseqs ``all_chains.m8`` hits file, if any.
        mmseqs_database: Local mmseqs database root, used to pre-populate PDB
            CIFs so Chai skips the RCSB download.

    Returns:
        ``(m8_path, store_dir)`` -- the m8 to pass to Chai (combined, or the
        original if nothing was added) and the store dir to serve CIFs from.
        Either may be ``None`` when there's nothing to do.
    """
    work_dir = Path(work_dir)
    store_dir = work_dir / "chai_template_store"
    store_dir.mkdir(parents=True, exist_ok=True)

    # 1. Pre-populate PDB hits from a local mmseqs database (best effort).
    prepopulated = 0
    if existing_m8 and Path(existing_m8).is_file() and mmseqs_database:
        prepopulated = _copy_local_pdb_cifs(
            Path(existing_m8), store_dir, Path(mmseqs_database)
        )
        if prepopulated:
            logger.info(
                "Pre-populated %d PDB template CIF(s) for Chai-1 from the local "
                "database (skipping RCSB downloads).",
                prepopulated,
            )

    # 2. Inject custom / inline templates as fabricated m8 hits.
    custom_rows: List[str] = []
    custom_idx = 0
    for seq in input_params.get("sequences", []):
        if "protein" not in seq:
            continue
        prot = seq["protein"]
        templates = prot.get("templates")
        if not templates:
            continue
        prot_ids = prot["id"] if isinstance(prot["id"], list) else [prot["id"]]
        for template in templates:
            mmcif = template.get("mmcif") if isinstance(template, dict) else None
            if not mmcif:
                continue
            synth_id = f"CT{custom_idx:02d}"
            custom_idx += 1
            chain_name = _normalise_and_gzip_cif(
                mmcif, store_dir / f"{synth_id}.cif.gz"
            )
            if chain_name is None:
                continue
            # One hit per query chain this template applies to.
            for prot_id in prot_ids:
                row = _fabricate_m8_row(
                    query_id=str(prot_id),
                    subject_id=f"{synth_id}_{chain_name}",
                    query_indices=template.get("queryIndices", []),
                    template_indices=template.get("templateIndices", []),
                )
                if row:
                    custom_rows.append(row)
    if custom_rows:
        logger.info(
            "Prepared %d custom template hit(s) for Chai-1.", len(custom_rows)
        )

    # 3. Assemble the m8 Chai will read.
    if not custom_rows and prepopulated == 0:
        # Nothing to add; caller can just use the original m8 (or none).
        return (Path(existing_m8) if existing_m8 else None), (
            store_dir if prepopulated else None
        )

    combined_m8 = work_dir / "chai_templates_combined.m8"
    lines: List[str] = []
    if existing_m8 and Path(existing_m8).is_file():
        lines.extend(
            line
            for line in Path(existing_m8).read_text().splitlines()
            if line.strip()
        )
    lines.extend(custom_rows)
    combined_m8.write_text("\n".join(lines) + "\n")

    return combined_m8, store_dir
