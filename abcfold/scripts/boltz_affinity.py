#!/usr/bin/env python
# mypy: ignore-errors
#
# Standalone Boltzina-style affinity scorer for ABCFold.
#
# Mirrors the ``ipsae`` script: it can be run directly on a single predicted
# complex (a CIF/PDB containing a protein with a placed ligand, as produced by
# AlphaFold3, Chai-1, Protenix, Boltz, ...) and writes the Boltz-2 affinity
# score for that complex.
#
# Unlike upstream Boltzina, no docking (AutoDock Vina) is performed: the ligand
# is already placed by the prediction back-end, so we go straight to the
# "scoring only" path -- parse the complex, build (or reuse) the Boltz
# ``processed`` work_dir, and run the Boltz-2 affinity head.
#
# Example
# -------
#   # auto-detect ligand, build processed inputs internally (needs net for MSA)
#   boltz_affinity model.cif --smiles "CC(=O)Oc1ccccc1C(=O)O" -o affinity.csv
#
#   # reuse a processed work_dir from an existing ABCFold/Boltz run
#   boltz_affinity model.cif --work_dir /path/to/boltz_run -o affinity.csv

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
    """Score a single predicted protein-ligand complex with Boltz-2 affinity.

    Args:
        input_model: Path to the complex structure (CIF or PDB) with the ligand
            already placed.
        output_dir: Working/output directory for intermediate Boltz files.
        work_dir: Optional existing Boltz ``processed`` work_dir to reuse. If
            ``None`` one is built internally from the structure.
        ligand_chain / ligand_resname: Optional overrides for ligand selection;
            by default the largest non-polymer heteromolecule is auto-detected.
        smiles / ccd: Ligand chemistry used when building processed inputs
            internally (one is required in that case).
        ligand_name: Residue/component name used for the ligand inside Boltz.
        seed / batch_size: Passed through to the affinity predictor.
        boltz_override: Recompute even if outputs already exist.
        use_msa_server: Use the MSA server when building processed inputs.
        clean_intermediate_files: Remove scratch files after scoring.
    """

    def __init__(
        self,
        input_model,
        output_dir,
        ligand_chain=None,
        ligand_resname=None,
        smiles=None,
        ccd=None,
        ligand_name="LIG",
        seed=None,
        use_msa_server=True,
        clean_intermediate_files=True,
        cache=None,
        mw_correction=False,
        msa=None,
    ):
        self.input_model = Path(input_model)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.mw_correction = mw_correction
        self.msa = Path(msa) if msa else None
        self.ligand_chain = ligand_chain
        self.ligand_resname = ligand_resname
        self.smiles = smiles
        self.ccd = ccd
        self.ligand_name = ligand_name
        self.cache = Path(cache) if cache else None
        self.seed = seed
        self.use_msa_server = use_msa_server
        self.clean_intermediate_files = clean_intermediate_files

        self.results = []

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
    def _load_structure(self):
        """Load the input structure into a BioPython model via CifFile."""
        from abcfold.output.file_handlers import CifFile

        path = self.input_model
        if path.suffix.lower() == ".pdb":
            # Convert PDB -> CIF so CifFile / parse_mmcif are happy.
            from Bio.PDB import MMCIFIO, PDBParser

            parser = PDBParser(QUIET=True)
            structure = parser.get_structure(path.stem, str(path))
            io = MMCIFIO()
            io.set_structure(structure)
            cif_path = self.output_dir / f"{path.stem}.cif"
            io.save(str(cif_path))
            self.complex_cif = cif_path
        else:
            self.complex_cif = path

        self.cif = CifFile(self.complex_cif)
        self.model = self.cif.model[0]
        self.record_id = self.input_model.stem

    # ------------------------------------------------------------------ #
    def _prepare_work_dir(self):
        """Build the processed work_dir (manifest, constraints, MSA) internally."""
        logger.info("Building Boltz processed inputs from %s", self.complex_cif)
        sequences = bu.extract_protein_sequences(self.model)
        if not sequences:
            raise ValueError("No protein chains found in the input structure.")
        if not (self.smiles or self.ccd):
            raise ValueError(
                "Building processed inputs requires --smiles or --ccd for the "
                "ligand."
            )

        yaml_path = bu.build_boltz_yaml(
            sequences=sequences,
            ligand_name=self.ligand_name,
            ligand_smiles=self.smiles,
            ligand_ccd=self.ccd,
            msa=self.msa,
            out_yaml=self.output_dir / f"{self.record_id}.yaml",
        )
        # A supplied MSA means there's nothing to fetch from the server.
        self.work_dir = bu.build_processed_inputs(
            yaml_path=yaml_path,
            work_dir=self.output_dir / f"boltz_work_{self.ligand_name}",
            cache_dir=self.cache,
            use_msa_server=self.use_msa_server and self.msa is None,
        )
        self.base_manifest = bu.load_manifest(self.work_dir)
        self.base_record_id = self.base_manifest["records"][0]["id"]

    # ------------------------------------------------------------------ #
    def run(self):

        self._load_structure()

        # Resolve which ligand to score.
        self.ligand = bu.select_ligand(
            self.model, self.ligand_chain, self.ligand_resname, smiles=self.smiles
        )

        # Canonical residue name shared by the CIF ligand, the mol pickles and
        # the Boltz manifest (mirrors Boltzina's base_ligand_name). Boltz names
        # SMILES ligands "LIG" (our default) and CCD ligands by their CCD code.
        if self.ccd:
            self.ligand_name = self.ccd.strip().upper()

        logger.info(
            "Scoring ligand %s (chain %s, %d atoms) as '%s'",
            self.ligand["resname"],
            self.ligand["chain_id"],
            self.ligand["num_atoms"],
            self.ligand_name,
        )

        # Namespace per-ligand artefacts so scoring different ligands of the
        # same model doesn't collide or reuse a stale work_dir/manifest.
        self.record_id = f"{self.input_model.stem}_{self.ligand_name}"

        self._prepare_work_dir()

        # One record / one pose for a single complex.
        record_ids = [self.record_id]

        # parse_mmcif parses the WHOLE complex and needs a mol for every non-CCD
        # residue, and the manifest only describes protein + the scored ligand.
        # So strip every other ligand / cofactor / ion (and waters).
        others = [
            (c["chain_id"], c["resseq"])
            for c in bu.detect_ligands(self.model, include_additives=True)
            if not (
                c["chain_id"] == self.ligand["chain_id"]
                and c["resseq"] == self.ligand["resseq"]
            )
        ]

        # Normalise: strip others, rename the scored ligand to the canonical
        # name, and regenerate entity/subchain records for boltzina's parser.
        self.scored_cif = bu.normalize_complex_cif(
            self.complex_cif,
            cif_out=self.output_dir / f"{self.record_id}_prepared.cif",
            ligand_chain=self.ligand["chain_id"],
            old_resname=self.ligand["resname"],
            new_resname=self.ligand_name,
            remove_residues=others,
        )

        # Ligand mol, written in BOTH layouts (datamodule + parse_mmcif).
        self.extra_mols_dir.mkdir(parents=True, exist_ok=True)
        self.parse_mols_dir.mkdir(parents=True, exist_ok=True)
        mol = bu.ligand_to_mol(
            self.model, self.ligand, smiles=self.smiles, work_dir=self.output_dir
        )
        bu.write_extra_mols(
            mol, record_ids, self.extra_mols_dir, ligand_name=self.ligand_name
        )
        bu.write_parse_mol(mol, self.ligand_name, self.parse_mols_dir)

        # SMILES ligands use our custom mol (drop the name from the CCD so the
        # parser uses ours); CCD ligands are resolved from the CCD itself.
        drop = None if self.ccd else self.ligand_name
        ccd = bu.load_ccd(cache_dir=self.cache, drop_name=drop)
        pose_dir = bu.prepare_affinity_structure(
            complex_cif=self.scored_cif,
            record_id=self.record_id,
            predictions_dir=self.predictions_dir,
            extra_mols_dir=self.parse_mols_dir,
            ccd=ccd,
            override=True,
        )
        if pose_dir is None:
            raise RuntimeError("Structure preparation failed; cannot score.")

        # Manifest + constraints for the scored record(s).
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

        # Remove any stale affinity output for these records so a skipped /
        # failed record reports as missing instead of re-reading an old result.
        self._clear_stale_affinity(record_ids)

        # Run the Boltz-2 affinity head.
        self._score()

        # Collect results.
        self.results = bu.extract_affinity_results(
            self.predictions_dir,
            record_ids,
            extra={
                "input_model": str(self.input_model),
                "ligand_chain": self.ligand["chain_id"],
                "ligand_resname": self.ligand["resname"],
            },
        )

        if self.clean_intermediate_files:
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
            "Boltzina-style Boltz-2 affinity scoring of a predicted "
            "protein-ligand complex (CIF/PDB)."
        ),
    )
    parser.add_argument(
        "input_model", help="Path to the complex structure (CIF or PDB)."
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        help="Working/output directory (default: alongside the input model).",
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
        "--ligand_resname",
        default=None,
        help="Override auto-detected ligand residue name.",
    )
    parser.add_argument(
        "--ligand_name",
        default="LIG",
        help="Component name to give the ligand inside Boltz (default: LIG).",
    )
    parser.add_argument("--seed", type=int, default=None, help="Random seed.")
    parser.add_argument(
        "--mw_correction",
        action="store_true",
        help=(
            "Apply Boltz's affinity molecular-weight correction to "
            "affinity_pred_value (off by default; match your native Boltz run)."
        ),
    )
    parser.add_argument(
        "--msa",
        default=None,
        help=(
            "Path to a precomputed MSA (.a3m) for the protein, reused instead of "
            "querying the MSA server. Faster and reproducible across runs; "
            "applied to all protein chains."
        ),
    )
    parser.add_argument(
        "--no_msa_server",
        action="store_true",
        help="Do not use the MSA server when building processed inputs.",
    )
    parser.add_argument(
        "--keep_intermediate",
        action="store_true",
        help="Keep intermediate Boltz files after scoring.",
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

    input_model = Path(args.input_model)
    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else input_model.parent / f"{input_model.stem}_boltz_affinity"
    )

    boltzina = Boltzina(
        input_model=input_model,
        output_dir=output_dir,
        ligand_chain=args.ligand_chain,
        ligand_resname=args.ligand_resname,
        smiles=args.smiles,
        ccd=args.ccd,
        ligand_name=args.ligand_name,
        seed=args.seed,
        use_msa_server=not args.no_msa_server,
        clean_intermediate_files=not args.keep_intermediate,
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
