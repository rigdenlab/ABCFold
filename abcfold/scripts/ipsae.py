#!/usr/bin/env python

import copy
import logging
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from Bio.PDB.Polypeptide import is_aa
from Bio.SeqUtils import seq1

from abcfold.output.file_handlers import (CifFile, ConfidenceJsonFile,
                                          FileTypes, NpyFile, NpzFile, PklFile,
                                          ResidueCountType)
from abcfold.output.utils import Af3Pae

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s"
)
logger = logging.getLogger("logger")


class Ipsae():
    def __init__(self,
                 input_model,
                 pae_file,
                 pae_cutoff=5,
                 pae_format="alphafold3",
                 distance_cutoff=10):
        self.input_model = input_model
        self.pae_file = pae_file
        self.pae_cutoff = pae_cutoff
        self.pae_format = pae_format
        self.distance_cutoff = distance_cutoff

        # Read structure + PAE data
        if isinstance(self.input_model, CifFile):
            self.struct = copy.deepcopy(self.input_model)
        else:
            self.struct = CifFile(self.input_model)
        self.process_input()

    @staticmethod
    def ptm_func(x, d0):
        x = np.asarray(x)
        d0 = float(d0)
        return 1.0 / (1.0 + np.square(x / d0))

    @staticmethod
    def calc_d0(L, pair_type='protein'):
        """
        Vectorized d0 calculator.

        - L : scalar or array-like of sequence lengths. Values < 27 are treated as 27.
        - pair_type : string. 'nucleic_acid' (case-insensitive) uses minimum d0 = 2.0;
                    any other value uses minimum d0 = 1.0.

        Returns a NumPy array (or scalar ndarray) of d0 values.
        """
        if not isinstance(pair_type, str):
            raise TypeError("pair_type must be a string")

        L = np.asarray(L, dtype=float)
        L = np.maximum(L, 27.0)

        min_value = 2.0 if pair_type.lower() == 'nucleic_acid' else 1.0
        d0 = 1.24 * np.power(L - 15.0, 1.0 / 3.0) - 1.8
        return np.maximum(min_value, d0)

    @staticmethod
    def nested_chain_map(default, chain_ids):
        return {c1: {c2: default() for c2 in chain_ids if c1 != c2} for c1 in chain_ids}

    @staticmethod
    def pdockq(plddt_mean, n_pairs):
        x = plddt_mean * np.log10(n_pairs)
        return 0.724 / (1 + np.exp(-0.052*(x - 152.611))) + 0.018

    @staticmethod
    def pdockq2(plddt_mean, mean_ptm):
        x2 = plddt_mean * mean_ptm
        return 1.31 / (1 + np.exp(-0.075*(x2 - 84.733))) + 0.005

    @property
    def chain_res_map(self):
        if not hasattr(self, '_chain_res_map'):
            self._chain_res_map = np.asarray([
                chain.id
                for chain in self.struct.get_chains()
                for res in chain if is_aa(res.resname, standard=False)
            ], dtype=object)
        return self._chain_res_map

    @property
    def nres(self):
        if not hasattr(self, '_nres'):
            self._nres = len(self.chain_res_map)
        return self._nres

    @property
    def chain_ids(self):
        if not hasattr(self, '_chain_ids'):
            self._chain_ids = sorted(set(self.chain_res_map))
        return self._chain_ids

    def process_input(self):
        """
        Process input files and setup data types
        """

        if isinstance(self.pae_file, ConfidenceJsonFile):
            file_ = self.pae_file
        else:
            # Find PAE data format
            suffix = Path(self.pae_file).suffix[1:]
            if suffix == FileTypes.NPZ.value:
                file_ = NpzFile(self.pae_file)
            elif suffix == FileTypes.NPY.value:
                file_ = NpyFile(self.pae_file)
            elif suffix == FileTypes.JSON.value:
                file_ = ConfidenceJsonFile(self.pae_file)
            elif suffix == FileTypes.PKL.value:
                file_ = PklFile(self.pae_file)
            else:
                raise ValueError(f"Unsupported PAE file type: {suffix}")

        # Construct input_params obj dict with protein seqs to avoid error msg
        self.struct.input_params = {"sequences": []}
        for chain in self.struct.get_chains():
            seq_data = {}
            seq_data["protein"] = {}
            if all([is_aa(res.resname) for res in chain.get_residues()]):
                seq_data["protein"]["ID"] = [chain.id]
                seq_data["protein"]["sequence"] = "".join(
                        [seq1(residue.get_resname()) for residue in chain]
                    )
                self.struct.input_params["sequences"].append(seq_data)

        # Get PAE data for different formats
        if self.pae_format == "alphafold2":
            self.pae_data = Af3Pae.from_alphafold2(
                file_.data,
                self.struct,
            ).scores
        elif self.pae_format == "alphafold3":
            self.pae_data = Af3Pae.from_alphafold3(
                file_.data,
                self.struct,
            ).scores
        elif self.pae_format == "boltz":
            self.pae_data = Af3Pae.from_boltz(
                file_.data,
                self.struct,
            ).scores
        elif self.pae_format == "chai":
            self.pae_data = Af3Pae.from_chai1(
                file_.data,
                self.struct,
            ).scores
        elif self.pae_format == "colabfold":
            self.pae_data = Af3Pae.from_colabfold(
                file_.data,
                self.struct,
            ).scores
        elif self.pae_format == "openfold3":
            self.pae_data = Af3Pae.from_openfold3(
                file_.data,
                self.struct,
            ).scores
        elif self.pae_format == "protenix":
            self.pae_data = Af3Pae.from_protenix(
                file_.data,
                self.struct,
            ).scores
        else:
            raise ValueError(f"Unsupported PAE format: {self.pae_format}")

        # Extract pLDDTs
        if "plddt" in self.pae_data:
            self.plddt = np.array(self.pae_data["plddt"])
        else:
            plddt = self.struct.get_plddt_per_residue(
                method=ResidueCountType.IPSAE.value
            )
            self.plddt = np.array(
                sum([plddt[chain] for chain in plddt.keys()
                     if not self.struct.check_ligand(chain)
                     ], [])
            )

        # Generate PAE matrix
        if 'pae' in self.pae_data:
            self.pae_matrix = np.array(self.pae_data['pae'])
        elif 'predicted_aligned_error' in self.pae_data:
            self.pae_matrix = np.array(self.pae_data['predicted_aligned_error'])
        else:
            self.pae_matrix = None
        if self.pae_format in ["alphafold3", "boltz", "chai"]:
            token_mask = self.get_token_mask()
            self.pae_matrix = self.pae_matrix[np.ix_(token_mask, token_mask)]

        return

    def get_token_mask(self):
        # Mask out non-standard atoms possible
        token_mask = []
        for chain in self.struct.get_chains():
            for residue in chain:
                if is_aa(residue.resname, standard=True):
                    if any((atom.id == "CA") or ("C1" in atom.id) for atom in residue):
                        token_mask.append(1)
                elif is_aa(residue.resname, standard=False):
                    for atom in residue:
                        if (atom.id == "CA") or ("C1" in atom.id):
                            token_mask.append(1)
                        else:
                            token_mask.append(0)
                elif self.struct.check_ligand(chain):
                    for atom in residue:
                        token_mask.append(0)
                else:
                    for atom in residue:
                        token_mask.append(0)
        return np.array(token_mask, dtype=bool)

    def get_cb_distance(self):
        cb_coords = []
        for chain in self.struct.get_chains():
            for residue in chain:
                chosen_atom = None
                for atom in residue:
                    if not is_aa(residue.resname, standard=False):
                        continue
                    if atom.id == "CB":
                        chosen_atom = atom
                        break
                    if "C3" in atom.id:
                        chosen_atom = atom
                        break
                    if residue.resname == "GLY" and atom.id == "CA":
                        chosen_atom = atom
                        break

                if chosen_atom:
                    cb_coords.append(chosen_atom.get_coord())

        cb_coords = np.array([x for x in cb_coords])
        distances = np.sqrt(
            ((cb_coords[:, None, :] - cb_coords[None, :, :])**2).sum(axis=2)
        )
        return distances

    def compute_pdockq(self, distances, cutoff=8.0):
        """
        Compute pDockQ and pDockQ2 for all inter-chain interfaces.

        Parameters
        ----------
        distances : (N, N) ndarray
            Matrix of residue–residue Cβ distances.
        cutoff : float
            Distance threshold for residue–residue contacts.

        Returns
        -------
        pdockq_scores, pdockq2_scores : dict[chain1][chain2] → float
        """

        if distances.shape[0] != self.nres or distances.shape[1] != self.nres:
            raise ValueError(
                f"Residue count mismatch: distances.shape={distances.shape}, \
                residues={self.nres}"
            )

        pdockq_scores = self.nested_chain_map(lambda: 0.0, self.chain_ids)
        pdockq2_scores = self.nested_chain_map(lambda: 0.0, self.chain_ids)
        unique_residues = self.nested_chain_map(set, self.chain_ids)

        pair_counts = self.nested_chain_map(lambda: 0, self.chain_ids)
        pae_ptm_sums = self.nested_chain_map(lambda: 0.0, self.chain_ids)

        for i in range(self.nres):
            c_i = self.chain_res_map[i]

            pc = pair_counts[c_i]
            ur = unique_residues[c_i]
            pps = pae_ptm_sums[c_i]

            close_js = np.where(distances[i] <= cutoff)[0]
            if close_js.size == 0:
                continue

            per_chain_js = defaultdict(list)
            for j in close_js:
                if j == i:
                    continue
                c_j = self.chain_res_map[j]
                if c_j == c_i:
                    continue
                per_chain_js[c_j].append(j)

            for c_j, js in per_chain_js.items():
                js_arr = np.asarray(js, dtype=int)

                n_pairs_here = js_arr.size

                pc[c_j] += n_pairs_here
                ur[c_j].add(i)
                ur[c_j].update(js_arr)

                pae_vals = self.pae_matrix[i, js_arr]
                ptm_vals = self.ptm_func(pae_vals, 10.0)
                pps[c_j] += ptm_vals.sum()

        for c1 in self.chain_ids:
            for c2 in self.chain_ids:
                if c1 == c2:
                    continue

                n_pairs = pair_counts[c1][c2]
                if n_pairs == 0:
                    pdockq_scores[c1][c2] = 0.0
                    pdockq2_scores[c1][c2] = 0.0
                    continue

                uniq_res_list = sorted(unique_residues[c1][c2])
                mean_plddt = float(
                    np.mean(self.plddt[uniq_res_list])
                    ) if len(uniq_res_list) > 0 else 0.0

                pdq = self.pdockq(mean_plddt, n_pairs)
                pdockq_scores[c1][c2] = float(np.round(pdq, 4))

                mean_ptm = pae_ptm_sums[c1][c2] / n_pairs
                pdq2 = self.pdockq2(mean_plddt, mean_ptm)
                pdockq2_scores[c1][c2] = float(np.round(pdq2, 4))
        return pdockq_scores, pdockq2_scores

    def compute_ligand_interface_scores(self):
        """
        Compute ligand interface scores (LIS) between all chain pairs
        using the PAE matrix and a PAE cutoff stored in self.pae_cutoff.
        """

        LIS = self.nested_chain_map(lambda: 0.0, self.chain_ids)

        for chain1 in self.chain_ids:
            for chain2 in self.chain_ids:
                if chain1 == chain2:
                    continue

                # Mask residues from chain1 vs chain2
                mask = (self.chain_res_map[:, None] == chain1) & \
                    (self.chain_res_map[None, :] == chain2)
                selected_pae = self.pae_matrix[mask]

                if selected_pae.size == 0:
                    LIS[chain1][chain2] = 0.0
                    continue

                # Apply cutoff
                valid_pae = selected_pae[selected_pae <= 12]
                if valid_pae.size == 0:
                    LIS[chain1][chain2] = 0.0
                    continue

                # Score = (cutoff - pae) / cutoff
                scores = (12 - valid_pae) / 12
                LIS[chain1][chain2] = float(np.round(np.mean(scores), 4))

        return LIS

    def compute_iptm_ipsae(self, distances):
        """
        Compute iPTM and IPSAE scores for all inter-chain interfaces.

        Parameters
        ----------
        distances : (N, N) ndarray
            Matrix of residue–residue Cβ distances.

        Returns
        -------
        dicts containing per-residue and per-chain-pair scores:
            iptm_d0chn_byres, ipsae_d0chn_byres,
            ipsae_d0dom_byres, ipsae_d0res_byres,
            iptm_d0chn_asym, ipsae_d0chn_asym,
            ipsae_d0dom_asym, ipsae_d0res_asym
        """

        # Initialize dictionaries
        iptm_d0chn_byres = self.nested_chain_map(
            lambda: np.zeros(self.nres), self.chain_ids)
        ipsae_d0chn_byres = self.nested_chain_map(
            lambda: np.zeros(self.nres), self.chain_ids)
        ipsae_d0dom_byres = self.nested_chain_map(
            lambda: np.zeros(self.nres), self.chain_ids)
        ipsae_d0res_byres = self.nested_chain_map(
            lambda: np.zeros(self.nres), self.chain_ids)

        iptm_d0chn_asym = self.nested_chain_map(lambda: 0.0, self.chain_ids)
        ipsae_d0chn_asym = self.nested_chain_map(lambda: 0.0, self.chain_ids)
        ipsae_d0dom_asym = self.nested_chain_map(lambda: 0.0, self.chain_ids)
        ipsae_d0res_asym = self.nested_chain_map(lambda: 0.0, self.chain_ids)

        # Track unique residues
        unique_residues_chain1 = self.nested_chain_map(set, self.chain_ids)
        unique_residues_chain2 = self.nested_chain_map(set, self.chain_ids)
        dist_unique_residues_chain1 = self.nested_chain_map(set, self.chain_ids)
        dist_unique_residues_chain2 = self.nested_chain_map(set, self.chain_ids)

        # Compute per-residue iPTM / IPSAE
        for chain1 in self.chain_ids:
            for chain2 in self.chain_ids:
                if chain1 == chain2:
                    continue

                n0chn = np.sum(self.chain_res_map == chain1) + \
                    np.sum(self.chain_res_map == chain2)
                d0chn = self.calc_d0(n0chn)

                ptm_matrix_d0chn = self.ptm_func(self.pae_matrix, d0chn)

                valid_pairs_iptm = (self.chain_res_map == chain2)
                valid_pairs_matrix = (self.chain_res_map == chain2) & \
                    (self.pae_matrix < self.pae_cutoff)

                for i in range(self.nres):
                    if self.chain_res_map[i] != chain1:
                        continue

                    # IPSAE / iPTM per residue
                    valid_pairs_ipsae = valid_pairs_matrix[i]
                    iptm_d0chn_byres[chain1][chain2][i] = (
                        ptm_matrix_d0chn[
                            i, valid_pairs_iptm
                        ].mean() if valid_pairs_iptm.any() else 0.0
                    )
                    ipsae_d0chn_byres[chain1][chain2][i] = (
                        ptm_matrix_d0chn[
                            i, valid_pairs_ipsae
                        ].mean() if valid_pairs_ipsae.any() else 0.0
                    )

                    # Track unique residues contributing
                    if valid_pairs_ipsae.any():
                        unique_residues_chain1[chain1][chain2].add(i)
                        for j in np.where(valid_pairs_ipsae)[0]:
                            unique_residues_chain2[chain1][chain2].add(j)

                    valid_pairs = (self.chain_res_map == chain2) & \
                        (self.pae_matrix[i] < self.pae_cutoff) & \
                        (distances[i] < self.distance_cutoff)
                    if valid_pairs.any():
                        dist_unique_residues_chain1[chain1][chain2].add(i)
                        for j in np.where(valid_pairs)[0]:
                            dist_unique_residues_chain2[chain1][chain2].add(j)

        for chain1 in self.chain_ids:
            for chain2 in self.chain_ids:
                if chain1 == chain2:
                    continue

                # Domain-level d0
                residues_1 = len(unique_residues_chain1[chain1][chain2])
                residues_2 = len(unique_residues_chain2[chain1][chain2])
                n0dom = residues_1 + residues_2
                d0dom = self.calc_d0(n0dom)

                # PTM matrix for domain-level IPSAE
                ptm_matrix_d0dom = self.ptm_func(self.pae_matrix, d0dom)

                # Valid pairs matrix
                mask_chain2 = (self.chain_res_map == chain2)
                mask_pae = (self.pae_matrix < self.pae_cutoff)
                valid_pairs_matrix = mask_pae & mask_chain2[None, :]

                # Per-residue d0 for IPSAE
                n0res_byres_all = np.sum(valid_pairs_matrix, axis=1)
                d0res_byres_all = self.calc_d0(n0res_byres_all)

                for i in range(self.nres):
                    if self.chain_res_map[i] != chain1:
                        continue

                    valid_pairs = valid_pairs_matrix[i]

                    ipsae_d0dom_byres[chain1][chain2][i] = ptm_matrix_d0dom[
                        i, valid_pairs
                    ].mean() if valid_pairs.any() else 0.0

                    ptm_row_d0res = self.ptm_func(
                        self.pae_matrix[i], d0res_byres_all[i]
                    )
                    ipsae_d0res_byres[chain1][chain2][i] = ptm_row_d0res[
                        valid_pairs
                    ].mean() if valid_pairs.any() else 0.0

        # Compute per-chain-pair iPTM / IPSAE as maximum over residues
        for c1 in self.chain_ids:
            for c2 in self.chain_ids:
                if c1 == c2:
                    continue

                iptm_d0chn_asym[c1][c2] = iptm_d0chn_byres[c1][c2].max()
                ipsae_d0chn_asym[c1][c2] = ipsae_d0chn_byres[c1][c2].max()
                ipsae_d0dom_asym[c1][c2] = ipsae_d0dom_byres[c1][c2].max()
                ipsae_d0res_asym[c1][c2] = ipsae_d0res_byres[c1][c2].max()

        return {
            "iptm_d0chn_asym": iptm_d0chn_asym,
            "ipsae_d0chn_asym": ipsae_d0chn_asym,
            "ipsae_d0dom_asym": ipsae_d0dom_asym,
            "ipsae_d0res_asym": ipsae_d0res_asym,
            "unique_residues_chain1": {
                c1: {c2: len(res_set) for c2, res_set in inner.items()}
                for c1, inner in unique_residues_chain1.items()
            },
            "unique_residues_chain2": {
                c1: {c2: len(res_set) for c2, res_set in inner.items()}
                for c1, inner in unique_residues_chain2.items()
            },
            "dist_unique_residues_chain1": {
                c1: {c2: len(res_set) for c2, res_set in inner.items()}
                for c1, inner in dist_unique_residues_chain1.items()
            },
            "dist_unique_residues_chain2": {
                c1: {c2: len(res_set) for c2, res_set in inner.items()}
                for c1, inner in dist_unique_residues_chain2.items()
            }
        }

    def main(self, output_csv=None, verbose=True):
        """
        Main function to calculate all the scores output by ipSAE
        """
        distances = self.get_cb_distance()

        pdockq_scores, pdockq2_scores = self.compute_pdockq(distances)
        ligand_interface_scores = self.compute_ligand_interface_scores()
        iptm_ipsae_scores = self.compute_iptm_ipsae(distances)

        self.output_results(pdockq_scores,
                            pdockq2_scores,
                            ligand_interface_scores,
                            iptm_ipsae_scores,
                            verbose,
                            output_csv)

        return {
            "pdockq": pdockq_scores,
            "pdockq2": pdockq2_scores,
            "ligand_interface_scores": ligand_interface_scores,
            "iptm_ipsae_scores": iptm_ipsae_scores
        }

    def output_results(self,
                       pdockq_scores,
                       pdockq2_scores,
                       lis_scores,
                       ipsae_scores,
                       verbose=True,
                       output_csv=None):

        if not verbose and output_csv is None:
            return

        metric_dicts = {}
        if ipsae_scores:
            metric_dicts["ipSAE"] = ipsae_scores['ipsae_d0res_asym']
            metric_dicts["ipSAE_d0chn"] = ipsae_scores['ipsae_d0chn_asym']
            metric_dicts["ipSAE_d0dom"] = ipsae_scores['ipsae_d0dom_asym']
            metric_dicts["iPTM_d0chn"] = ipsae_scores['iptm_d0chn_asym']
        if pdockq_scores:
            metric_dicts["pDockQ"] = pdockq_scores
        if pdockq2_scores:
            metric_dicts["pDockQ2"] = pdockq2_scores
        if lis_scores:
            metric_dicts["LIS"] = lis_scores
        if ipsae_scores:
            metric_dicts['nres1'] = ipsae_scores['unique_residues_chain1']
            metric_dicts['nres2'] = ipsae_scores['unique_residues_chain2']
            metric_dicts['dist1'] = ipsae_scores['dist_unique_residues_chain1']
            metric_dicts['dist2'] = ipsae_scores['dist_unique_residues_chain2']

        rows = []
        chain_pairs = set()
        for metric, scores in metric_dicts.items():
            if scores is None:
                continue
            for ch1, inner in scores.items():
                for ch2 in inner.keys():
                    chain_pairs.add((ch1, ch2))

        # Build rows for each pair
        for ch1, ch2 in sorted(chain_pairs):
            row = {"Chn1": ch1, "Chn2": ch2}

            # Add available metrics
            for metric, scores in metric_dicts.items():
                if scores is None:
                    continue
                try:
                    row[metric] = scores[ch1][ch2]
                except KeyError:
                    row[metric] = None  # missing entry → leave blank

            rows.append(row)

        # Convert to DataFrame
        df = pd.DataFrame(rows)

        if verbose:
            logger.info(df.to_string(index=False))

        if output_csv:
            df.to_csv(output_csv, index=False)

        return df


def main():
    import argparse

    from Bio.PDB import MMCIFIO, PDBParser

    parser = argparse.ArgumentParser(
        prog="ipsae",
        description="Compute IPSAE metrics from structure and PAE file"
    )

    parser.add_argument("input_model", help="Path to input structure (PDB/mmCIF)")
    parser.add_argument("pae_file", help="Path to PAE file (npz/npy/json)")
    parser.add_argument("--pae_cutoff", type=float, default=10.0,
                        help="PAE cutoff used in some scores (default: 10.0)")
    parser.add_argument("--pae_format",
                        choices=[
                            "alphafold2", "alphafold3", "boltz",
                            "chai", "colabfold", "opefold3", "protenix"
                        ],
                        default="alphafold3",
                        help="Format of the PAE file (default: alphafold3)")
    parser.add_argument("--distance_cutoff", type=float, default=10.0,
                        help="Cβ distance cutoff in Ångström (default: 10.0)")
    parser.add_argument("--quiet", "-q", action="store_true",
                        help="Enable verbose logging/output")
    parser.add_argument("--output", "-o", type=Path,
                        help="Optional output CSV path for results")

    args = parser.parse_args()

    tmp_cif = None
    pdb_path = Path(args.input_model)
    if pdb_path.suffix == ".pdb":
        pdb_parser = PDBParser(QUIET=True)
        structure = pdb_parser.get_structure(pdb_path.stem, str(pdb_path))
        io = MMCIFIO()
        io.set_structure(structure)
        tmp_cif = pdb_path.with_suffix(".cif")
        io.save(str(tmp_cif))
        args.input_model = tmp_cif

    ipsae_calculator = Ipsae(
        args.input_model,
        args.pae_file,
        args.pae_cutoff,
        args.pae_format,
        args.distance_cutoff
    )

    if args.quiet:
        verbose = False
    else:
        verbose = True

    ipsae_calculator.main(verbose=verbose, output_csv=args.output)

    if tmp_cif is not None:
        try:
            Path(tmp_cif).unlink()
        except FileNotFoundError:
            pass


if __name__ == "__main__":
    main()
