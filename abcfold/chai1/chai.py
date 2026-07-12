# Copyright (c) 2024 Chai Discovery, Inc.
# Licensed under the Apache License, Version 2.0.
# See the LICENSE file for details.

# Notice: This file has been modified to include a wrapper around the
# `run_inference` function to allow for the PAE (Predicted Aligned Error)
# to be output. The wrapper function captures the PAE output and
# integrates it into the command line interface.

"""Command line interface."""

import logging
import shutil
from pathlib import Path

import numpy as np
import typer
from chai_lab.chai1 import run_inference

logging.basicConfig(level=logging.INFO)


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
        _install_template_store(Path(template_cif_store))

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
