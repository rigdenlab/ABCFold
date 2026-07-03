"""Batch Boltz-2 affinity scoring of predicted models for ABCFold.

``boltz_affinity`` needs the boltz micromamba env + GPU, so it can't run
in-process in the base ABCFold env. This helper invokes the ``boltz_affinity``
multi-model scorer as a subprocess -- it self-dispatches into the boltz env,
scores every model against one shared MSA in a single inference pass, and
writes a combined CSV. We read that back into a ``{model_path: scores}`` lookup
the HTML/report code can join against per model.
"""

import logging
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Union

logger = logging.getLogger("logger")


def extract_affinity_msa(
    input_params: dict, output_dir: Union[str, Path]
) -> Optional[str]:
    """Return an .a3m path to reuse for affinity scoring, or None to fetch.

    Reuses the MSA ABCFold already built (mmseqs2, or a user-supplied
    ``unpairedMsa`` / ``unpairedMsaPath`` in the input) so the affinity step
    doesn't re-query the MSA server. Only done for a *single* protein sequence:
    one alignment can't be shared across different chains in a heteromer, and
    ``--msa`` applies it to all protein chains.
    """
    proteins = [
        s["protein"]
        for s in input_params.get("sequences", [])
        if isinstance(s, dict) and "protein" in s
    ]
    if len(proteins) != 1:
        return None

    prot = proteins[0]
    if prot.get("unpairedMsaPath"):
        return str(prot["unpairedMsaPath"])
    if prot.get("unpairedMsa"):
        msa_file = Path(output_dir) / "affinity" / "reused_msa.a3m"
        msa_file.parent.mkdir(parents=True, exist_ok=True)
        msa_file.write_text(prot["unpairedMsa"])
        logger.info("Reusing ABCFold's MSA for affinity scoring: %s", msa_file)
        return str(msa_file)
    return None


def run_boltz_affinity(
    model_paths: List[Union[str, Path]],
    output_dir: Union[str, Path],
    smiles: Optional[str] = None,
    ccd: Optional[str] = None,
    ligand_chain: Optional[str] = None,
    msa: Optional[Union[str, Path]] = None,
) -> Dict[str, dict]:
    """Score every model with the Boltz-2 affinity head in one batched pass.

    Args:
        model_paths: Predicted complex structures (CIF/PDB), all the same
            protein + ligand.
        output_dir: Where to put the ``affinity/`` working dir and results CSV.
        smiles / ccd: Ligand chemistry (one is required to score).
        ligand_chain: Optional override for which chain holds the ligand.
        msa: Optional precomputed ``.a3m`` reused across all models.

    Returns:
        ``{resolved_model_path: {"affinity_pred_value": float,
        "affinity_probability_binary": float}}``. Empty on failure or when no
        ligand chemistry is given, so callers can proceed without scores.
    """
    paths = [Path(p).resolve() for p in model_paths]
    if not paths:
        return {}
    if not (smiles or ccd):
        logger.info(
            "No ligand SMILES/CCD provided; skipping affinity scoring."
        )
        return {}

    affinity_dir = Path(output_dir) / "affinity"
    cmd = [
        sys.executable,
        "-m",
        "abcfold.scripts.boltz_affinity",
        *[str(p) for p in paths],
        "--output_dir",
        str(affinity_dir),
    ]
    if smiles:
        cmd += ["--smiles", smiles]
    elif ccd:
        cmd += ["--ccd", ccd]
    if ligand_chain:
        cmd += ["--ligand_chain", ligand_chain]
    if msa:
        cmd += ["--msa", str(msa)]

    logger.info("Running Boltz-2 affinity scoring on %d model(s)...", len(paths))
    try:
        # Capture the (verbose) subprocess output; only surface it on failure.
        result = subprocess.run(cmd, capture_output=True, text=True)
    except FileNotFoundError as exc:
        logger.error("Could not launch affinity scoring: %s", exc)
        return {}
    if result.returncode != 0:
        tail = "\n".join((result.stderr or result.stdout or "").splitlines()[-25:])
        logger.error("Affinity scoring failed:\n%s", tail)
        return {}

    results_csv = affinity_dir / "boltz_affinity_results.csv"
    if not results_csv.exists():
        logger.warning("No affinity results were produced at %s", results_csv)
        return {}

    import pandas as pd

    df = pd.read_csv(results_csv)
    scores: Dict[str, dict] = {}
    for _, row in df.iterrows():
        model = str(row.get("input_model", ""))
        if not model:
            continue
        key = str(Path(model).resolve())
        scores[key] = {
            "affinity_pred_value": row.get("affinity_pred_value"),
            "affinity_probability_binary": row.get(
                "affinity_probability_binary"
            ),
        }
    return scores
