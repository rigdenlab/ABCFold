#!/usr/bin/env python
# mypy: ignore-errors
#
# Standalone Boltzina-style affinity scorer for ABCFold.
#
# Mirrors the ``ipsae`` script: run it on one or more predicted complexes
# (CIF/PDB containing a protein with a placed ligand, as produced by
# AlphaFold3, Chai-1, Protenix, Boltz, ...) and it writes the Boltz-2 affinity
# score for each.
#
# Unlike upstream Boltzina, no docking (AutoDock Vina) is performed: the ligand
# is already placed by the prediction back-end, so we go straight to the
# "scoring only" path -- parse the complex, build the Boltz ``processed``
# work_dir, and run the Boltz-2 affinity head. With several models of the same
# protein+ligand the MSA is built once and reused, and all poses are scored in
# one batched pass into a single CSV.
#
# Example
# -------
#   # one model
#   boltz_affinity model.cif --smiles "CC(=O)Oc1ccccc1C(=O)O"
#
#   # many models of the same target, sharing one MSA
#   boltz_affinity preds/*_model_*.cif --smiles "CC(=O)Oc1ccccc1C(=O)O" \
#       --msa target.a3m

import argparse
import configparser
import importlib.util
import json
import logging
import os
import sys
from pathlib import Path

import pandas as pd

from abcfold.affinity import boltzina_utils as bu

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("logger")

DEFAULT_BOLTZ_ENV = "abcfold-boltz-py311"


def _boltz_available() -> bool:
    """True if boltz + boltzina are importable in the current interpreter."""
    return (
        importlib.util.find_spec("boltz") is not None
        and importlib.util.find_spec("boltzina") is not None
    )


def _resolve_boltz_env(explicit: "str | None" = None) -> str:
    """Resolve the micromamba env that has boltz/boltzina installed.

    Precedence: explicit ``--boltz_env`` > ``$ABCFOLD_BOLTZ_ENV`` > the
    ``boltz_env`` key in the ABCFold config (~/.abcfold_config.ini, falling back
    to the bundled ``data/config.ini``) > a sensible default.
    """
    if explicit:
        return explicit
    env = os.environ.get("ABCFOLD_BOLTZ_ENV")
    if env:
        return env
    user_cfg = Path.home() / ".abcfold_config.ini"
    bundled = Path(__file__).resolve().parents[1] / "data" / "config.ini"
    cfg_path = user_cfg if user_cfg.exists() else bundled
    try:
        parser = configparser.ConfigParser()
        parser.read(str(cfg_path))
        for section in parser.sections():
            if parser.has_option(section, "boltz_env"):
                return parser.get(section, "boltz_env")
    except Exception:  # noqa: BLE001
        pass
    return DEFAULT_BOLTZ_ENV


def _dispatch_to_boltz_env(boltz_env: str) -> int:
    """Re-execute this command inside the boltz micromamba env.

    boltz/boltzina/torch only live in the boltz env, so when invoked from an env
    without them (e.g. the base ``abcfold`` env) we hand the whole run off to the
    boltz env -- mirroring how ``run_boltz.py`` shells out via ``MicromambaEnv``.
    The repo is put on ``PYTHONPATH`` so abcfold is importable in the child
    without being installed there, and ``--_in_boltz_env`` stops it re-dispatching.
    """
    from abcfold.backend_envs import MicromambaEnv

    repo_root = Path(__file__).resolve().parents[2]
    child_env = os.environ.copy()
    existing = child_env.get("PYTHONPATH", "")
    child_env["PYTHONPATH"] = (
        str(repo_root) + (os.pathsep + existing if existing else "")
    )

    passthrough = [a for a in sys.argv[1:]]
    cmd = [
        "python",
        "-m",
        "abcfold.scripts.boltz_affinity",
        *passthrough,
        "--_in_boltz_env",
    ]
    logger.info("Dispatching to boltz env '%s'...", boltz_env)
    micromamba = MicromambaEnv(boltz_env)
    full_cmd = [micromamba.micromamba, "run", "-n", boltz_env, *cmd]
    import subprocess

    return subprocess.call(full_cmd, env=child_env)


class Boltzina:
    """Score one or more predicted protein-ligand complexes with Boltz-2.

    Every model must be the same protein + ligand (e.g. different predicted
    poses); the MSA/manifest are built once and each pose is parsed and scored
    against them in a single batched affinity pass.

    Args:
        input_models: One path, or a list of paths, to complex structures
            (CIF/PDB) with the ligand already placed.
        output_dir: Working/output directory for intermediate Boltz files and
            the combined results.
        ligand_chain: Optional override restricting ligand selection to one
            chain; by default the ligand is auto-detected (SMILES-matched).
        smiles / ccd: Ligand chemistry used when building processed inputs
            (one is required).
        msa: Optional precomputed MSA (.a3m) used instead of querying the MSA
            server -- reuse one MSA to remove MSA-driven score variance and skip
            the fetch.
        mw_correction: Apply Boltz's affinity molecular-weight correction (off
            by default, matching native Boltz).
        seed: Seed for MSA subsampling; fixed by default for reproducibility.
    """

    def __init__(
        self,
        input_models,
        output_dir,
        ligand_chain=None,
        smiles=None,
        ccd=None,
        seed=42,
        cache=None,
        mw_correction=False,
        msa=None,
    ):
        if isinstance(input_models, (str, Path)):
            input_models = [input_models]
        self.input_models = [Path(m) for m in input_models]
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.mw_correction = mw_correction
        self.msa = Path(msa) if msa else None
        self.ligand_chain = ligand_chain
        self.smiles = smiles
        self.ccd = ccd
        # One canonical ligand name shared by every model (same protein+ligand):
        # the CCD code for --ccd, else "LIG" (what Boltz names a SMILES ligand).
        self.ligand_name = self.ccd.strip().upper() if self.ccd else "LIG"
        self.cache = Path(cache) if cache else None
        self.seed = seed

        self.results = []
        self.model_meta = {}

        # Boltz output layout under output_dir.
        self.boltz_out = self.output_dir / "boltz_out"
        self.predictions_dir = self.boltz_out / "predictions"
        self.processed_dir = self.boltz_out / "processed"
        # Two distinct mol-pickle layouts are required:
        #   * extra_mols_dir : {record_id}.pkl -> {ligand_name: mol}  (datamodule)
        #   * parse_mols_dir : {ligand_name}.pkl -> mol               (parse_mmcif)
        self.extra_mols_dir = self.processed_dir / "mols"
        self.parse_mols_dir = self.boltz_out / "parse_mols"
        self.constraints_dir = self.processed_dir / "constraints"

    # ------------------------------------------------------------------ #
    def _load_structure(self, path):
        """Load a structure into a BioPython model; return (model, cif_path)."""
        from abcfold.output.file_handlers import CifFile

        path = Path(path)
        if path.suffix.lower() == ".pdb":
            # Convert PDB -> CIF so CifFile / parse_mmcif are happy.
            from Bio.PDB import MMCIFIO, PDBParser

            parser = PDBParser(QUIET=True)
            structure = parser.get_structure(path.stem, str(path))
            io = MMCIFIO()
            io.set_structure(structure)
            cif_path = self.output_dir / f"{path.stem}.cif"
            io.save(str(cif_path))
        else:
            cif_path = path

        return CifFile(cif_path).model[0], cif_path

    # ------------------------------------------------------------------ #
    def _build_shared_inputs(self):
        """Build the processed work_dir (MSA, manifest) once for all models.

        Every model is the same protein + ligand, so the MSA/manifest are built
        a single time from the first model and reused for the whole batch.
        """
        if not (self.smiles or self.ccd):
            raise ValueError(
                "Scoring requires --smiles or --ccd for the ligand."
            )
        model, _ = self._load_structure(self.input_models[0])
        sequences = bu.extract_protein_sequences(model)
        if not sequences:
            raise ValueError("No protein chains found in the input structure.")

        logger.info("Building Boltz processed inputs (MSA, manifest)...")
        yaml_path = bu.build_boltz_yaml(
            sequences=sequences,
            ligand_name=self.ligand_name,
            ligand_smiles=self.smiles,
            ligand_ccd=self.ccd,
            msa=self.msa,
            out_yaml=self.output_dir / "boltz_affinity.yaml",
        )
        self.work_dir = bu.build_processed_inputs(
            yaml_path=yaml_path,
            work_dir=self.output_dir / f"boltz_work_{self.ligand_name}",
            cache_dir=self.cache,
            use_msa_server=self.msa is None,
        )
        self.base_manifest = bu.load_manifest(self.work_dir)
        self.base_record_id = self.base_manifest["records"][0]["id"]
        self.extra_mols_dir.mkdir(parents=True, exist_ok=True)
        self.parse_mols_dir.mkdir(parents=True, exist_ok=True)
        # CCD ligands resolve from the CCD; SMILES ligands use our custom mol so
        # the name is dropped from the CCD to force the parser to use ours.
        self.ccd_components = bu.load_ccd(
            cache_dir=self.cache,
            drop_name=None if self.ccd else self.ligand_name,
        )

    # ------------------------------------------------------------------ #
    def _prepare_model(self, model_path, index=0):
        """Parse one model, prepare its affinity structure. Returns record id."""
        model_path = Path(model_path)
        model, complex_cif = self._load_structure(model_path)
        ligand = bu.select_ligand(model, self.ligand_chain, smiles=self.smiles)
        # Index-prefixed so models sharing a stem (e.g. many "model.cif" in
        # different seed dirs) get unique records.
        record_id = f"m{index:03d}_{model_path.stem}_{self.ligand_name}"
        logger.info(
            "Preparing %s: ligand %s (chain %s) as '%s'",
            model_path.name, ligand["resname"], ligand["chain_id"],
            self.ligand_name,
        )

        # Strip every other ligand/cofactor/ion so the parsed structure matches
        # the manifest (protein + the scored ligand only).
        others = [
            (c["chain_id"], c["resseq"])
            for c in bu.detect_ligands(model, include_additives=True)
            if not (
                c["chain_id"] == ligand["chain_id"]
                and c["resseq"] == ligand["resseq"]
            )
        ]
        scored_cif = bu.normalize_complex_cif(
            complex_cif,
            cif_out=self.output_dir / f"{record_id}_prepared.cif",
            ligand_chain=ligand["chain_id"],
            old_resname=ligand["resname"],
            new_resname=self.ligand_name,
            remove_residues=others,
        )

        mol = bu.ligand_to_mol(
            model, ligand, smiles=self.smiles, work_dir=self.output_dir
        )
        bu.write_extra_mols(
            mol, [record_id], self.extra_mols_dir, ligand_name=self.ligand_name
        )
        bu.write_parse_mol(mol, self.ligand_name, self.parse_mols_dir)

        pose_dir = bu.prepare_affinity_structure(
            complex_cif=scored_cif,
            record_id=record_id,
            predictions_dir=self.predictions_dir,
            extra_mols_dir=self.parse_mols_dir,
            ccd=self.ccd_components,
            override=True,
        )
        if pose_dir is None:
            logger.warning("Skipping %s: structure preparation failed.",
                           model_path.name)
            return None

        self.model_meta[record_id] = {
            "input_model": str(model_path),
            "ligand_chain": ligand["chain_id"],
            "ligand_resname": ligand["resname"],
        }
        return record_id

    # ------------------------------------------------------------------ #
    def run(self):
        self._build_shared_inputs()

        # Prepare every model's structure; collect the ones that succeeded.
        record_ids = []
        for index, model_path in enumerate(self.input_models):
            rid = self._prepare_model(model_path, index)
            if rid is not None:
                record_ids.append(rid)
        if not record_ids:
            raise RuntimeError("No models could be prepared for scoring.")

        # One manifest + constraints covering all records, then a single
        # batched affinity pass over them.
        bu.build_record_manifest(
            base_manifest=self.base_manifest,
            base_record_id=self.base_record_id,
            record_ids=record_ids,
            out_manifest=self.processed_dir / "manifest.json",
        )
        bu.link_constraints(
            source_work_dir=self.work_dir,
            base_record_id=self.base_record_id,
            record_ids=record_ids,
            target_constraints_dir=self.constraints_dir,
        )
        self._clear_stale_affinity(record_ids)

        logger.info("Scoring %d model(s) with the Boltz-2 affinity head...",
                    len(record_ids))
        self._score()

        results = bu.extract_affinity_results(self.predictions_dir, record_ids)
        for r in results:
            r.update(self.model_meta.get(r.get("record_id"), {}))
        self.results = results

        self._cleanup()
        return self.results

    # ------------------------------------------------------------------ #
    def _clear_stale_affinity(self, record_ids):
        """Delete pre-existing affinity JSON for these records before scoring.

        Without this, a record the affinity step skips (e.g. the cropper
        failing) would silently re-read a previous run's result and report it
        as if fresh.
        """
        for rid in record_ids:
            stale = self.predictions_dir / rid / f"affinity_{rid}.json"
            if stale.exists():
                try:
                    stale.unlink()
                except OSError:
                    pass

    # ------------------------------------------------------------------ #
    def _score(self):
        """Invoke the Boltz-2 affinity predictor on the prepared pose."""
        from abcfold.affinity.predict_affinity import (load_boltz2_model,
                                                       predict_affinity)

        logger.info("Scoring pose(s) with Boltz-2 affinity head...")
        # skip_run_structure=True scores the input pose as-is (the point of the
        # tool); confidence prediction isn't needed in that path.
        model_module = load_boltz2_model(
            skip_run_structure=True,
            run_trunk_and_structure=True,
            affinity_mw_correction=self.mw_correction,
        )
        predict_affinity(
            self.work_dir,
            model_module=model_module,
            output_dir=str(self.predictions_dir),
            structures_dir=str(self.predictions_dir),
            constraints_dir=str(self.constraints_dir),
            extra_mols_dir=self.extra_mols_dir,
            manifest_path=self.processed_dir / "manifest.json",
            num_workers=1,
            batch_size=1,
            seed=self.seed,
        )

    # ------------------------------------------------------------------ #
    def _cleanup(self):
        for pre in self.predictions_dir.glob("*/pre_affinity_*.npz"):
            affinity = pre.parent / pre.name.replace(
                "pre_affinity_", "affinity_"
            ).replace(".npz", ".json")
            if affinity.exists():
                pre.unlink(missing_ok=True)

    # ------------------------------------------------------------------ #
    def get_results_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(self.results)

    def save_results(self, output_path):
        output_path = Path(output_path)
        if not self.results:
            logger.warning("No results to save.")
            return
        if output_path.suffix.lower() == ".json":
            with open(output_path, "w") as fh:
                json.dump(self.results, fh, indent=2)
        else:
            self.get_results_dataframe().to_csv(output_path, index=False)
        logger.info("Results saved to %s", output_path)


def main():
    parser = argparse.ArgumentParser(
        prog="boltz_affinity",
        description=(
            "Boltzina-style Boltz-2 affinity scoring of one or more predicted "
            "protein-ligand complexes (CIF/PDB). Multiple models of the same "
            "protein+ligand are scored against a shared MSA and written to one "
            "combined CSV."
        ),
    )
    parser.add_argument(
        "input_models",
        nargs="+",
        help=(
            "One or more complex structures (CIF/PDB), or a glob like "
            "'preds/*_model_*.cif'. All must be the same protein+ligand; the "
            "MSA is built once and each model scored against it."
        ),
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        help="Working/output directory (default: alongside the input models).",
    )
    parser.add_argument(
        "--cache",
        default=None,
        help=(
            "Boltz cache directory (holds ccd.pkl, mols/, weights). Defaults to "
            "$BOLTZ_CACHE or ~/.boltz. Point this at an already-populated cache "
            "to avoid re-downloading."
        ),
    )
    parser.add_argument(
        "--smiles",
        default=None,
        help="Ligand SMILES (used to build processed inputs / bond orders).",
    )
    parser.add_argument(
        "--ccd",
        default=None,
        help="Ligand CCD code (alternative to --smiles when building inputs).",
    )
    parser.add_argument(
        "--ligand_chain",
        default=None,
        help="Override auto-detected ligand chain id.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help=(
            "Random seed for MSA subsampling (default: 42). Fixed by default so "
            "scores are reproducible; pass a different value or vary it to "
            "sample the affinity distribution."
        ),
    )
    parser.add_argument(
        "--mw_correction",
        action="store_true",
        help=(
            "Apply Boltz's affinity molecular-weight correction (off by "
            "default, matching native Boltz)."
        ),
    )
    parser.add_argument(
        "--msa",
        default=None,
        help=(
            "Path to a precomputed MSA (.a3m) for the protein, used instead of "
            "querying the MSA server. Reuse one MSA across many models of the "
            "same protein to remove MSA-driven score variance (and skip the "
            "fetch). Applied to all protein chains."
        ),
    )
    parser.add_argument(
        "--boltz_env",
        default=None,
        help=(
            "Name of the micromamba env containing boltz/boltzina. Defaults to "
            "$ABCFOLD_BOLTZ_ENV, the config 'boltz_env' key, or "
            f"'{DEFAULT_BOLTZ_ENV}'. Only used when boltz isn't already "
            "importable (i.e. when run from the base abcfold env)."
        ),
    )
    # Internal flag: set when we have already re-executed inside the boltz env,
    # to prevent an infinite dispatch loop.
    parser.add_argument(
        "--_in_boltz_env", action="store_true", help=argparse.SUPPRESS
    )
    args = parser.parse_args()

    # If boltz/boltzina aren't importable here, hand the whole run off to the
    # boltz micromamba env (unless we're already running inside it).
    if not args._in_boltz_env and not _boltz_available():
        return _dispatch_to_boltz_env(_resolve_boltz_env(args.boltz_env))

    # Expand any globs the shell didn't (e.g. quoted patterns) and de-dupe.
    import glob as _glob

    input_models = []
    for pattern in args.input_models:
        matches = _glob.glob(pattern)
        input_models.extend(matches if matches else [pattern])
    input_models = [Path(m) for m in dict.fromkeys(input_models)]
    missing = [m for m in input_models if not m.exists()]
    if missing:
        parser.error(f"input model(s) not found: {[str(m) for m in missing]}")

    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else input_models[0].parent / "boltz_affinity"
    )

    boltzina = Boltzina(
        input_models=input_models,
        output_dir=output_dir,
        ligand_chain=args.ligand_chain,
        smiles=args.smiles,
        ccd=args.ccd,
        seed=args.seed,
        cache=args.cache,
        mw_correction=args.mw_correction,
        msa=args.msa,
    )

    boltzina.run()

    boltzina.save_results(output_dir / "boltz_affinity_results.csv")

    df = boltzina.get_results_dataframe()
    if not df.empty:
        logger.info("\n%s", df.to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
