import json
from pathlib import Path
from typing import Union

import numpy as np

from abcfold.processoutput.file_handlers import CifFile

AF3TEMPLATE: dict = {
    "atom_chain_ids": [],
    "atom_plddts": [],
    "contact_probs": [],
    "pae": [],
    "token_chain_ids": [],
    "token_res_ids": [],
}


class Af3Pae:

    @classmethod
    def from_boltz1(cls, scores: dict, cif_file: CifFile):
        af3_scores = AF3TEMPLATE.copy()

        chain_lengths = cif_file.chain_lengths(mode="residues", ligand_atoms=True)
        residue_lengths = cif_file.chain_lengths(mode="all", ligand_atoms=True)

        atom_chain_ids = flatten(
            [[key] * value for key, value in residue_lengths.items()]
        )

        atom_plddts = cif_file.plddts
        token_chain_ids = flatten(
            [[key] * value for key, value in chain_lengths.items()]
        )

        token_res_ids = flatten(
            [
                [value for value in values]
                for _, values in cif_file.token_residue_ids().items()
            ]
        )

        af3_scores["pae"] = scores["pae"].tolist()
        af3_scores["atom_chain_ids"] = atom_chain_ids
        af3_scores["atom_plddts"] = atom_plddts
        af3_scores["contact_probs"] = np.zeros(shape=scores["pae"].shape).tolist()
        af3_scores["token_chain_ids"] = token_chain_ids
        af3_scores["token_res_ids"] = token_res_ids

        return cls(af3_scores)

    @classmethod
    def from_chai1(cls, scores: np.ndarray, cif_file: CifFile):
        af3_scores = AF3TEMPLATE.copy()
        chain_lengths = cif_file.chain_lengths(mode="residues", ligand_atoms=True)

        residue_lengths = cif_file.chain_lengths(mode="all", ligand_atoms=True)

        atom_chain_ids = flatten(
            [[key] * value for key, value in residue_lengths.items()]
        )

        atom_plddts = cif_file.plddts
        token_chain_ids = flatten(
            [[key] * value for key, value in chain_lengths.items()]
        )

        token_res_ids = flatten(
            [
                [value for value in values]
                for _, values in cif_file.token_residue_ids().items()
            ]
        )

        af3_scores["pae"] = scores.tolist()
        af3_scores["atom_chain_ids"] = atom_chain_ids
        af3_scores["atom_plddts"] = atom_plddts
        af3_scores["contact_probs"] = np.zeros(shape=scores.shape).tolist()
        af3_scores["token_chain_ids"] = token_chain_ids
        af3_scores["token_res_ids"] = token_res_ids

        return cls(af3_scores)

    def __init__(self, af3_scores: dict):
        self.scores = af3_scores

    def to_file(self, file_path: Union[str, Path]):
        with open(file_path, "w") as f:
            json.dump(self.scores, f, indent=2)


def flatten(xss):
    return [x for xs in xss for x in xs]
