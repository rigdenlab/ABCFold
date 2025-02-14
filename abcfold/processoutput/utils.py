import json
from itertools import zip_longest
from pathlib import Path
from typing import List, Union

import numpy as np
from Bio.Align import PairwiseAligner

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


def get_gap_indicies(*cif_objs) -> List[np.ndarray]:
    """
    Get the the gaps inbetween cif objects. Sometimes there is a discrepency
    between chain lengths between the modelling programs. This function is
    used to find where these discrepencies are.

    Args:
        *cif_objs: Multiple cif objects

    Returns:
        indicies: Dict with the chain_id as the key where the discrepency is located and
            the value is a list of indicies with -1 representing gaps

    """
    indicies: list = []

    if len(cif_objs) == 1:
        return indicies
    chain_lengths = [
        cif.chain_lengths(mode="residues", ligand_atoms=True) for cif in cif_objs
    ]

    assert all(
        [
            chain_lengths[0].keys() == chain_lengths[i].keys()
            for i in range(1, len(chain_lengths) - 1)
        ]
    )

    unequal_chain_lengths = [
        id_
        for id_ in chain_lengths[0].keys()
        if any(
            [
                chain_lengths[0][id_] != chain_lengths[i][id_]
                for (i, _) in enumerate(chain_lengths[1:], start=1)
            ]
        )
    ]

    for chain_id in chain_lengths[0]:
        if chain_id in unequal_chain_lengths:
            # indicies[chain_id] = []
            chain_atoms = [
                "".join([atom.element for atom in cif.get_atoms(chain_id=chain_id)])
                for cif in cif_objs
            ]

            longest = max(chain_atoms, key=len)

            for atom_str in chain_atoms:
                alignment = PairwiseAligner().align(longest, atom_str)
                indicies.append(alignment[0].indices[1])
        else:
            for _ in cif_objs:

                indicies.append(np.array([1] * chain_lengths[0][chain_id]))

    indicies = interleave_repeated(
        indicies, len(cif_objs), len(list(chain_lengths[0].keys()))
    )

    return indicies


def interleave_repeated(lst, n, chain_no):
    indicies = []
    chunks = [lst[i:i+n] for i in range(0, len(lst), n)]
    interleaved = [x for tup in zip_longest(*chunks) for x in tup if x is not None]

    for i in range(0, len(interleaved), chain_no):
        tmp_lst = []
        for j in range(chain_no):
            tmp_lst.extend(interleaved[i + j])
        indicies.append(tmp_lst)

    return indicies


def insert_none_by_minus_one(indices, values):
    result = []
    value_index = 0

    for idx in indices:
        if idx == -1:
            result.append(None)
        else:
            result.append(values[value_index])
            value_index += 1

    assert len(indices) == len(result)

    return result


# WIP - local testing purposes

# if __name__ == "__main__":
#     input_params = {
#         "name": "Hello_fold",
#         "modelSeeds": [42],
#         "sequences": [
#             {
#                 "protein": {
#                     "id": "A",
#                     "sequence": "PVLSCGEWQL",
#                     "modifications": [
#                         {"ptmType": "HY3", "ptmPosition": 1},
#                         {"ptmType": "P1L", "ptmPosition": 5},
#                     ],
#                 }
#             },
#             {"protein": {"id": "B", "sequence": "QIQLVQSGPELKKPGET"}},
#             {"protein": {"id": "C", "sequence": "DVLMIQTPLSLPVS"}},
#             {"ligand": {"id": ["F"], "ccdCodes": ["ATP"]}},
#             {"ligand": {"id": "I", "ccdCodes": ["NAG", "FUC", "FUC"]}},
#             {"dna": {"id": ["D", "K"], "sequence": "AGCT"}},
#             {"rna": {"id": "L", "sequence": "AGCU"}},
#             {"ligand": {"id": "Z", "smiles": "CC(=O)OC1C[NH+]2CCC1CC2"}},
#         ],
#         "bondedAtomPairs": [
#             [["A", 1, "CA"], ["B", 1, "CA"]],
#             [["C", 7, "CA"], ["A", 10, "CA"]],
#             [["I", 1, "O3"], ["I", 2, "C1"]],
#             [["I", 2, "C1"], ["I", 3, "C1"]],
#         ],
#         "dialect": "alphafold3",
#         "version": 2,
#     }

#     # cif_file = CifFile(
#     #     "/home/etk48667/folding/aaa_bbb_ccc/chai1_Hello_fold/pred.model_idx_0.cif",
#     #     input_params,
#     # )
#     cif_file = CifFile(
#         "/home/etk48667/folding/drtest/alphafold3_Hello_fold/seed-42_sample-0/m\
# odel.cif",
#         input_params,
#     )

#     cif_file.pathway = "tmp.cif"

#     print(cif_file.chain_lengths(mode="residues", ligand_atoms=True))
