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
#   boltzina model.cif --smiles "CC(=O)Oc1ccccc1C(=O)O" -o affinity.csv
#
#   # reuse a processed work_dir from an existing ABCFold/Boltz run
#   boltzina model.cif --work_dir /path/to/boltz_run -o affinity.csv

import argparse
import json
import logging
from pathlib import Path

import pandas as pd

from abcfold.affinity import boltzina_utils as bu

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("logger")


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
        work_dir=None,
        ligand_chain=None,
        ligand_resname=None,
        smiles=None,
        ccd=None,
        ligand_name="LIG",
        seed=None,
        batch_size=1,
        boltz_override=False,
        use_msa_server=True,
        clean_intermediate_files=True,
        cache=None,
    ):
        self.input_model = Path(input_model)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.external_work_dir = Path(work_dir) if work_dir else None
        self.ligand_chain = ligand_chain
        self.ligand_resname = ligand_resname
        self.smiles = smiles
        self.ccd = ccd
        self.ligand_name = ligand_name
        self.cache = Path(cache) if cache else None
        self.seed = seed
        self.batch_size = batch_size
        self.boltz_override = boltz_override
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
        """Build the processed work_dir internally, or reuse the external one."""
        if self.external_work_dir is not None:
            logger.info("Reusing Boltz work_dir: %s", self.external_work_dir)
            self.work_dir = self.external_work_dir
            self.base_manifest = bu.load_manifest(self.work_dir)
            self.base_record_id = self.base_manifest["records"][0]["id"]
            return

        logger.info("Building Boltz processed inputs from %s", self.complex_cif)
        sequences = bu.extract_protein_sequences(self.model)
        if not sequences:
            raise ValueError("No protein chains found in the input structure.")
        if not (self.smiles or self.ccd):
            raise ValueError(
                "Building processed inputs internally requires --smiles or "
                "--ccd for the ligand. Alternatively pass --work_dir from an "
                "existing Boltz run."
            )

        yaml_path = bu.build_boltz_yaml(
            sequences=sequences,
            ligand_name=self.ligand_name,
            ligand_smiles=self.smiles,
            ligand_ccd=self.ccd,
            out_yaml=self.output_dir / f"{self.record_id}.yaml",
        )
        self.work_dir = bu.build_processed_inputs(
            yaml_path=yaml_path,
            work_dir=self.output_dir / "boltz_work",
            cache_dir=self.cache,
            use_msa_server=self.use_msa_server,
        )
        self.base_manifest = bu.load_manifest(self.work_dir)
        self.base_record_id = self.base_manifest["records"][0]["id"]

    # ------------------------------------------------------------------ #
    def run(self):
        self._load_structure()

        # Resolve ligand and build its RDKit mol.
        self.ligand = bu.select_ligand(
            self.model, self.ligand_chain, self.ligand_resname
        )
        logger.info(
            "Scoring ligand %s (chain %s, %d atoms)",
            self.ligand["resname"],
            self.ligand["chain_id"],
            self.ligand["num_atoms"],
        )

        self._prepare_work_dir()

        # One record / one pose for a single complex.
        record_ids = [self.record_id]

        # Normalise the predicted complex CIF for boltzina's parser: regenerate
        # entity/subchain records (gemmi setup_entities) AND give the ligand the
        # single canonical residue name shared by the mol pickles and manifest
        # (mirrors Boltzina's base_ligand_name). Predictors name the ligand
        # arbitrarily (e.g. LIG0), so the rename is restricted to its chain.
        original_resname = self.ligand["resname"]
        self.scored_cif = bu.normalize_complex_cif(
            self.complex_cif,
            cif_out=self.output_dir / f"{self.record_id}_prepared.cif",
            ligand_chain=self.ligand["chain_id"],
            old_resname=original_resname,
            new_resname=self.ligand_name,
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

        # Per-record CCD + structure preparation (parse the relabeled CIF using
        # the parse_mmcif mol layout).
        ccd = bu.load_ccd(cache_dir=self.cache, drop_name=self.ligand_name)
        pose_dir = bu.prepare_affinity_structure(
            complex_cif=self.scored_cif,
            record_id=self.record_id,
            predictions_dir=self.predictions_dir,
            extra_mols_dir=self.parse_mols_dir,
            ccd=ccd,
            override=self.boltz_override,
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
    def _score(self):
        """Invoke the Boltz-2 affinity predictor."""
        from abcfold.affinity.predict_affinity import (load_boltz2_model,
                                                       predict_affinity)

        logger.info("Scoring pose(s) with Boltz-2 affinity head...")
        model_module = load_boltz2_model(
            skip_run_structure=True,
            run_trunk_and_structure=True,
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
            batch_size=self.batch_size,
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
        prog="boltzina",
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
        "--work_dir",
        default=None,
        help=(
            "Existing Boltz 'processed' work_dir to reuse (e.g. from an ABCFold "
            "Boltz run). If omitted, processed inputs are built internally."
        ),
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
        "--batch_size", type=int, default=1, help="Affinity batch size."
    )
    parser.add_argument(
        "--no_msa_server",
        action="store_true",
        help="Do not use the MSA server when building processed inputs.",
    )
    parser.add_argument(
        "--boltz_override",
        action="store_true",
        help="Recompute even if intermediate outputs already exist.",
    )
    parser.add_argument(
        "--keep_intermediate",
        action="store_true",
        help="Keep intermediate Boltz files after scoring.",
    )
    parser.add_argument(
        "--output",
        "-o",
        default=None,
        help="Output path for results (.csv or .json).",
    )
    args = parser.parse_args()

    input_model = Path(args.input_model)
    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else input_model.parent / f"{input_model.stem}_boltzina"
    )

    boltzina = Boltzina(
        input_model=input_model,
        output_dir=output_dir,
        work_dir=args.work_dir,
        ligand_chain=args.ligand_chain,
        ligand_resname=args.ligand_resname,
        smiles=args.smiles,
        ccd=args.ccd,
        ligand_name=args.ligand_name,
        seed=args.seed,
        batch_size=args.batch_size,
        boltz_override=args.boltz_override,
        use_msa_server=not args.no_msa_server,
        clean_intermediate_files=not args.keep_intermediate,
        cache=args.cache,
    )

    boltzina.run()

    output_path = (
        Path(args.output)
        if args.output
        else output_dir / "boltzina_results.csv"
    )
    boltzina.save_results(output_path)

    df = boltzina.get_results_dataframe()
    if not df.empty:
        logger.info("\n%s", df.to_string(index=False))


if __name__ == "__main__":
    main()
