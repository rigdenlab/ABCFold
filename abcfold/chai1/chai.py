# Copyright (c) 2024 Chai Discovery, Inc.
# Licensed under the Apache License, Version 2.0.
# See the LICENSE file for details.

# Notice: This file has been modified to include a wrapper around the
# `run_inference` function to allow for the PAE (Predicted Aligned Error)
# to be output. The wrapper function captures the PAE output and
# integrates it into the command line interface.

"""Command line interface."""

import json
import logging
import shutil
from pathlib import Path

import numpy as np
import typer
from chai_lab.chai1 import run_inference

logging.basicConfig(level=logging.INFO)


def _read_fasta_proteins(fasta_file: Path) -> dict:
    """Return {chain_id: sequence} for the ``>protein|<id>`` records."""
    seqs: dict = {}
    chain = None
    for line in Path(fasta_file).read_text().splitlines():
        if line.startswith(">"):
            parts = line[1:].split("|")
            chain = parts[1] if len(parts) > 1 and parts[0] == "protein" else None
        elif chain is not None and line.strip():
            seqs[chain] = seqs.get(chain, "") + line.strip()
    return seqs


def _subject_span(query_seq: str, template_seq: str, kalign_fn):
    """Template [start, end) that aligns to the query, via chai-lab's kalign.

    chai-lab validates each hit by re-aligning the template slice to the query
    with kalign and checking the subject span. We compute that span here with
    the *same* kalign so the m8 we emit is consistent by construction. Returns
    a 0-indexed half-open span into ``template_seq``, or None if no alignment.
    """
    alignment = kalign_fn(ref=query_seq, query=template_seq)
    if alignment is None:
        return None
    t_idx = -1
    first = last = None
    # reference_aligned is the query; query_aligned is the template.
    for q_ch, t_ch in zip(alignment.reference_aligned, alignment.query_aligned):
        if t_ch != "-":
            t_idx += 1
            if q_ch != "-":  # template residue aligned to a query residue
                if first is None:
                    first = t_idx
                last = t_idx
    if first is None or last is None:
        return None
    return first, last + 1


def _build_custom_m8(
    store: Path, fasta_file: Path, existing_m8: Path | None, out_m8: Path
):
    """Assemble the Chai m8, adding kalign-derived rows for custom templates.

    Reads the custom-template manifest staged by ABCFold, aligns each template
    to its query chain(s) with chai-lab's kalign, and appends m8 rows whose
    subject spans match that alignment (so chai-lab's TemplateHit validation
    passes). Returns the m8 to use, or None if there's nothing to pass.
    """
    import gemmi

    rows: list[str] = []
    manifest_path = store / "custom_templates.json"
    if manifest_path.is_file():
        from chai_lab.tools.kalign import kalign_query_to_reference

        manifest = json.loads(manifest_path.read_text())
        queries = _read_fasta_proteins(fasta_file)
        for synth_id, info in manifest.items():
            cif_gz = store / f"{synth_id}.cif.gz"
            if not cif_gz.is_file():
                continue
            try:
                structure = gemmi.read_structure(str(cif_gz))
                tchain = info["template_chain"]
                tseq = (
                    structure[0][tchain]
                    .get_polymer()
                    .make_one_letter_sequence()
                    .replace("-", "")
                )
            except Exception as exc:  # noqa: BLE001
                logging.warning("Custom template %s unreadable: %s", synth_id, exc)
                continue
            for qchain in info.get("query_chains", []):
                qseq = queries.get(qchain)
                if not qseq:
                    continue
                span = _subject_span(qseq, tseq, kalign_query_to_reference)
                if span is None:
                    logging.warning(
                        "Custom template %s did not align to chain %s; skipping",
                        synth_id,
                        qchain,
                    )
                    continue
                s_start, s_end = span
                rows.append(
                    "\t".join(
                        [
                            str(qchain),                 # query_id
                            f"{synth_id}_{tchain}",      # subject_id
                            "100.0",                     # pident
                            str(s_end - s_start),        # length
                            "0", "0",                    # mismatch, gapopen
                            "1", str(len(qseq)),         # query start/end (unused)
                            str(s_start + 1), str(s_end),  # subject start/end
                            "1e-9", "100.0", "custom",   # evalue, bitscore, comment
                        ]
                    )
                )
        if rows:
            logging.info("Added %d custom template hit(s) to the m8", len(rows))

    lines: list[str] = []
    if existing_m8 is not None and Path(existing_m8).is_file():
        lines += [
            ln for ln in Path(existing_m8).read_text().splitlines() if ln.strip()
        ]
    lines += rows
    if not lines:
        return None
    out_m8.write_text("\n".join(lines) + "\n")
    return out_m8


def _install_template_store(store: Path) -> None:
    """Serve template CIFs from a local store instead of downloading from RCSB.

    chai-lab fetches each template as ``{PDBID}.cif.gz`` via
    ``chai_lab.data.io.rcsb.download_cif_file`` (which skips the download when
    the file already exists). We monkeypatch that function so that, when a
    matching ``{PDBID}.cif.gz`` exists in ``store``, it is copied into Chai's
    per-run cache folder and returned -- no network call. This both avoids
    re-downloading PDB templates ABCFold already has and lets custom (non-PDB)
    templates be served (their ids would 404 against RCSB). Anything not in the
    store falls back to the original download behaviour.
    """
    import chai_lab.data.io.rcsb as rcsb

    original = rcsb.download_cif_file

    def _patched(pdb_id: str, directory: Path) -> Path:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        outfile = directory / f"{pdb_id}.cif.gz"
        if not outfile.exists():
            local = store / f"{pdb_id}.cif.gz"
            if local.exists():
                shutil.copyfile(local, outfile)
                logging.info("Using local template CIF for %s", pdb_id)
                return outfile
        return original(pdb_id, directory)

    rcsb.download_cif_file = _patched


CITATION = """
@article{Chai-1-Technical-Report,
    title        = {Chai-1: Decoding the molecular interactions of life},
    author       = {{Chai Discovery}},
    year         = 2024,
    journal      = {bioRxiv},
    publisher    = {Cold Spring Harbor Laboratory},
    doi          = {10.1101/2024.10.10.615955},
    url          = {https://www.biorxiv.org/content/early/2024/10/11/2024.10.10.615955},
    elocation-id = {2024.10.10.615955},
    eprint       = {https://www.biorxiv.org/content/early/2024/10/11/2024.10.1\
0.615955.full.pdf}
}
""".strip()


def citation():
    """Print citation information"""
    typer.echo(CITATION)


def run_inference_wrapper(
    fasta_file: Path,
    *,
    output_dir: Path,
    use_esm_embeddings: bool = True,
    use_msa_server: bool = False,
    msa_server_url: str = "https://api.colabfold.com",
    msa_directory: Path | None = None,
    constraint_path: Path | None = None,
    use_templates_server: bool = False,
    template_hits_path: Path | None = None,
    template_cif_store: Path | None = None,
    # Parameters controlling how we do inference
    recycle_msa_subsample: int = 0,
    num_trunk_recycles: int = 3,
    num_diffn_timesteps: int = 200,
    num_diffn_samples: int = 5,
    num_trunk_samples: int = 1,
    seed: int | None = None,
    device: str | None = None,
    low_memory: bool = True,
):

    if template_cif_store is not None and Path(template_cif_store).is_dir():
        store = Path(template_cif_store)
        _install_template_store(store)
        # Build the final m8 here (in the Chai env) so custom-template subject
        # spans are computed with chai-lab's own kalign.
        combined = _build_custom_m8(
            store, fasta_file, template_hits_path, store / "chai_combined.m8"
        )
        if combined is not None:
            template_hits_path = combined

    result = run_inference(
        fasta_file=fasta_file,
        output_dir=output_dir,
        use_esm_embeddings=use_esm_embeddings,
        use_msa_server=use_msa_server,
        msa_server_url=msa_server_url,
        msa_directory=msa_directory,
        constraint_path=constraint_path,
        use_templates_server=use_templates_server,
        template_hits_path=template_hits_path,
        recycle_msa_subsample=recycle_msa_subsample,
        num_trunk_recycles=num_trunk_recycles,
        num_diffn_timesteps=num_diffn_timesteps,
        num_diffn_samples=num_diffn_samples,
        num_trunk_samples=num_trunk_samples,
        seed=seed,
        device=device,
        low_memory=low_memory,
    )

    np.save(f"{output_dir}/pae_scores.npy", result.pae)
    return result


def cli():
    app = typer.Typer()
    app.command("fold", help="Run Chai-1 to fold a complex.")(run_inference_wrapper)
    app.command("citation", help="Print citation information")(citation)
    app()


if __name__ == "__main__":
    cli()
