import configparser
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
    def from_alphafold2(cls, scores: dict, cif_file: CifFile):
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

        af3_scores["pae"] = scores["predicted_aligned_error"].tolist()
        af3_scores["atom_chain_ids"] = atom_chain_ids
        af3_scores["atom_plddts"] = atom_plddts
        af3_scores["contact_probs"] = np.zeros(
            shape=scores["predicted_aligned_error"].shape
        ).tolist()
        af3_scores["token_chain_ids"] = token_chain_ids
        af3_scores["token_res_ids"] = token_res_ids

        return cls(af3_scores)

    @classmethod
    def from_alphafold3(cls, scores: dict, cif_file: CifFile):
        def reorder_matrix(pae_matrix, chain_lengths, af3_chain_lengths):
            if not isinstance(pae_matrix, np.ndarray):
                pae_matrix = np.array(pae_matrix)
            desired_order = flatten(
                [[key] * value for key, value in chain_lengths.items()]
            )
            current_order = flatten(
                [[key] * value for key, value in af3_chain_lengths.items()]
            )

            order_mapping = {}

            for i, chain_id in enumerate(current_order):
                if chain_id in desired_order:
                    order_mapping[i] = desired_order.index(chain_id)
                    desired_order[desired_order.index(chain_id)] = None  # Mark as used
                else:
                    order_mapping[i] = i

            # Reorder the PAE matrix rows and columns based on the mapping

            reordered_matrix = np.zeros_like(pae_matrix)
            for i in range(len(pae_matrix)):
                for j in range(len(pae_matrix)):

                    reordered_matrix[order_mapping[i], order_mapping[j]] = pae_matrix[
                        i, j
                    ]

            return reordered_matrix.tolist()

        af3_scores = AF3TEMPLATE.copy()

        chain_lengths = cif_file.chain_lengths(
            mode="residues", ligand_atoms=True, ptm_atoms=True
        )

        af3pae_chain_lengths = {
            k: chain_lengths[k] for k in np.unique(scores["token_chain_ids"])
        }
        if list(chain_lengths.keys()) == list(af3pae_chain_lengths.keys()):
            return cls(scores)

        residue_lengths = cif_file.chain_lengths(mode="all", ligand_atoms=True)

        atom_chain_ids = flatten(
            [[key] * value for key, value in residue_lengths.items()]
        )
        atom_plddts = cif_file.plddts

        token_res_ids_dict = cif_file.token_residue_ids()
        token_res_ids = flatten([value for _, value in token_res_ids_dict.items()])

        reordered_pae = reorder_matrix(
            scores["pae"], chain_lengths, af3pae_chain_lengths
        )
        contact_probs = reorder_matrix(
            scores["contact_probs"], chain_lengths, af3pae_chain_lengths
        )
        token_chain_ids = flatten(
            [[key] * len(value) for key, value in token_res_ids_dict.items()]
        )

        assert len(atom_chain_ids) == len(scores["atom_chain_ids"])
        assert sum([len(value) for value in token_res_ids_dict.values()]) == len(
            scores["pae"]
        )
        assert len(atom_plddts) == len(scores["atom_plddts"])

        assert len(reordered_pae) == len(scores["pae"])
        assert len(contact_probs) == len(scores["contact_probs"])
        assert len(reordered_pae[0]) == len(scores["pae"][0])
        assert len(contact_probs[0]) == len(scores["contact_probs"][0])
        assert len(token_chain_ids) == len(scores["token_chain_ids"])
        assert len(token_res_ids) == len(scores["token_res_ids"])

        af3_scores["atom_chain_ids"] = atom_chain_ids
        af3_scores["atom_plddts"] = atom_plddts
        af3_scores["contact_probs"] = contact_probs
        af3_scores["pae"] = reordered_pae
        af3_scores["token_chain_ids"] = token_chain_ids
        af3_scores["token_res_ids"] = token_res_ids

        return cls(af3_scores)

    @classmethod
    def from_boltz(cls, scores: dict, cif_file: CifFile):
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

    @classmethod
    def from_colabfold(cls, scores: dict, cif_file: CifFile):
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

        pae_matrix = np.asarray(scores["pae"])
        af3_scores["pae"] = pae_matrix.tolist()
        af3_scores["atom_chain_ids"] = atom_chain_ids
        af3_scores["atom_plddts"] = atom_plddts
        af3_scores["contact_probs"] = np.zeros(shape=pae_matrix.shape).tolist()
        af3_scores["token_chain_ids"] = token_chain_ids
        af3_scores["token_res_ids"] = token_res_ids

        return cls(af3_scores)

    @classmethod
    def from_protenix(cls, scores: dict, cif_file: CifFile):
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

        pae_matrix = np.asarray(scores["token_pair_pae"])
        af3_scores["pae"] = pae_matrix.tolist()
        af3_scores["atom_chain_ids"] = atom_chain_ids
        af3_scores["atom_plddts"] = atom_plddts
        af3_scores["contact_probs"] = np.zeros(shape=pae_matrix.shape).tolist()
        af3_scores["token_chain_ids"] = token_chain_ids
        af3_scores["token_res_ids"] = token_res_ids

        return cls(af3_scores)

    @classmethod
    def from_openfold3(cls, scores: dict, cif_file: CifFile):
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

        pae_matrix = np.asarray(scores["pde"])
        af3_scores["pae"] = pae_matrix.tolist()
        af3_scores["atom_chain_ids"] = atom_chain_ids
        af3_scores["atom_plddts"] = atom_plddts
        af3_scores["contact_probs"] = np.zeros(shape=pae_matrix.shape).tolist()
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


def verify_config_file(abc_config, default_config_file):
    """
    Verify that the config file has the correct keys and values

    Args:
        abc_config (Union[str, Path]): The config file to verify
        default_config_file (Union[str, Path]): The path to the default config file

    Returns:
        None
    """

    config_a = configparser.SafeConfigParser()
    config_a.read(str(abc_config))

    config_b = configparser.SafeConfigParser()
    config_b.read(str(default_config_file))

    for section in config_b.sections():
        if section == "Versions":
            # Update the versions to match the default config file so we know we're
            # using compatible versions of the programs
            config_a[section] = config_b[section]
        if not config_a.has_section(section):
            config_a.add_section(section)
        for key in config_b[section]:
            if not config_a.has_option(section, key):
                config_a[section][key] = config_b[section][key]

    with open(abc_config, "w") as f:
        config_a.write(f)
