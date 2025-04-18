import json
from itertools import zip_longest
from pathlib import Path
from typing import List, Union

import numpy as np
import pandas as pd
from Bio.Align import PairwiseAligner

from abcfold.output.file_handlers import CifFile

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
    def from_alphafold3(cls, scores: dict, cif_file: CifFile):
        def reorder_matrix(pae_matrix, chain_lengths, af3_chain_lengths):
            if not isinstance(pae_matrix, np.ndarray):
                pae_matrix = np.array(pae_matrix)
            desired = flatten([[k] * v for k, v in chain_lengths.items()])
            current = flatten([[k] * v for k, v in af3_chain_lengths.items()])
            order = {}
            for i, c in enumerate(current):
                if c in desired:
                    idx = desired.index(c)
                    order[i] = idx
                    desired[idx] = None
                else:
                    order[i] = i
            out = np.zeros_like(pae_matrix)
            for i in range(len(pae_matrix)):
                for j in range(len(pae_matrix)):
                    out[order[i], order[j]] = pae_matrix[i, j]
            return out.tolist()

        af3_scores = AF3TEMPLATE.copy()
        chain_lengths = cif_file.chain_lengths(mode="residues", ligand_atoms=True, ptm_atoms=True)

        token_chains = np.unique(scores["token_chain_ids"])
        missing = set(token_chains) - set(chain_lengths.keys())
        if missing and "L" in chain_lengths:
            json_ids = [ch.id for ch in cif_file.get_chains()]
            mapping = {chr(ord("A")+i): json_ids[i] for i in range(len(json_ids))}
            scores["token_chain_ids"] = [mapping.get(c, c) for c in scores["token_chain_ids"]]
            token_chains = np.unique(scores["token_chain_ids"])

        af3pae_chain_lengths = {k: chain_lengths[k] for k in token_chains}
        if list(chain_lengths.keys()) == list(af3pae_chain_lengths.keys()):
            return cls(scores)

        residue_lengths = cif_file.chain_lengths(mode="all", ligand_atoms=True)
        atom_chain_ids = flatten([[k] * v for k, v in residue_lengths.items()])
        atom_plddts = cif_file.plddts
        token_res = flatten(list(cif_file.token_residue_ids().values()))

        reordered_pae = reorder_matrix(scores["pae"], chain_lengths, af3pae_chain_lengths)
        contact = reorder_matrix(scores["contact_probs"], chain_lengths, af3pae_chain_lengths)
        token_chain_ids = flatten([[k] * len(v) for k, v in cif_file.token_residue_ids().items()])

        af3_scores["atom_chain_ids"] = atom_chain_ids
        af3_scores["atom_plddts"] = atom_plddts
        af3_scores["contact_probs"] = contact
        af3_scores["pae"]           = reordered_pae
        af3_scores["token_chain_ids"] = token_chain_ids
        af3_scores["token_res_ids"] = token_res

        return cls(af3_scores)


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
            json.dump(self.scores, f, indent=4)


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
    chunks = [lst[i : i + n] for i in range(0, len(lst), n)]  # noqa: E203
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


def make_dummy_m8_file(run_json, output_dir):
    """
    Make a dummy m8 file with the templates from the run JSON file
    """
    with open(run_json) as f:
        input_json = json.load(f)

    templates = {}
    for sequence in input_json["sequences"]:
        if "protein" not in sequence:
            continue
        for id_ in sequence["protein"]["id"]:
            for template in sequence["protein"]["templates"]:
                if id_ in templates:
                    templates[id_].append(
                        template["mmcif"].split("\n")[0].split("_")[1]
                    )
                else:
                    templates[id_] = [template["mmcif"].split("\n")[0].split("_")[1]]

    m8_file = output_dir / "dummy.m8"
    if not templates:
        return None
    table = []

    for id_ in templates:
        for template in templates[id_]:
            table.append([id_, template, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    pd.DataFrame(table).to_csv(m8_file, sep="\t", header=False, index=False)

    return m8_file
