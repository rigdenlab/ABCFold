import json
import logging
import random
import string
from pathlib import Path
from typing import Dict, List, Optional, Union

DELIM = "      "
logger = logging.getLogger("logger")


class BoltzYaml:
    """
    Convert AlphaFold3-style JSON to Boltz YAML.
    """

    def __init__(self, working_dir: Union[str, Path], create_files: bool = True):
        self.working_dir = working_dir
        self.yaml_string: str = ""
        self.msa_file: Optional[Union[str, Path]] = "null"
        self.__ids: List[Union[str, int]] = []
        self.__id_char: str = "A"
        self.__id_links: Dict[Union[str, int], list] = {}
        self.__create_files = create_files
        self.__non_ligands: List[str] = []
        self.__id_buffer: dict = {}

    @property
    def chain_ids(self) -> List[Union[str, int]]:
        return self.__ids

    @property
    def id_links(self) -> Dict[Union[str, int], list]:
        return self.__id_links

    def msa_to_file(self, msa: str, file_path: Union[str, Path]):
        with open(file_path, "w") as f:
            f.write(msa)

    def __merge_duplicate_sequences(self, sequences: list) -> list:
        merged: Dict[str, dict] = {}
        leftovers: List[dict] = []
        for entry in sequences:
            if "protein" in entry:
                seq = entry["protein"]["sequence"]
                current_id = entry["protein"]["id"]
                if seq in merged:
                    existing = merged[seq]["protein"]
                    ids = existing["id"]
                    if not isinstance(ids, list):
                        ids = [ids]
                    ids.extend(current_id if isinstance(current_id, list) else [current_id])
                    existing["id"] = ids
                else:
                    merged[seq] = entry
            else:
                leftovers.append(entry)
        return leftovers + list(merged.values())

    def json_to_yaml(self, json_file_or_dict: Union[dict, str, Path]):
        if isinstance(json_file_or_dict, (str, Path)):
            json_dict = json.loads(Path(json_file_or_dict).read_text())
        else:
            json_dict = json_file_or_dict

        sequences = self.__merge_duplicate_sequences(json_dict["sequences"])
        self.get_ids(sequences)

        self.yaml_string = ""
        bonded_atom_string = ""
        self.yaml_string += self.add_version_number("1")

        for key, value in json_dict.items():
            if key == "sequences":
                if "sequences" not in self.yaml_string:
                    self.yaml_string += self.add_non_indented_string("sequences")
                for sequence_dict in sequences:
                    if any(k in sequence_dict for k in ["protein", "rna", "dna"]):
                        self.yaml_string += self.sequence_to_yaml(sequence_dict)
                    if "ligand" in sequence_dict:
                        self.yaml_string += self.add_ligand_information(sequence_dict["ligand"])
            if key == "bondedAtomPairs" and isinstance(value, list):
                bonded_atom_string += self.bonded_atom_pairs_to_yaml(value)
                if "constraints" not in self.yaml_string and bonded_atom_string:
                    self.yaml_string += self.add_non_indented_string("constraints")
                self.yaml_string += bonded_atom_string
        return self.yaml_string

    def bonded_atom_pairs_to_yaml(self, bonded_atom_pairs: list):
        yaml_string = ""
        for pair in bonded_atom_pairs:
            if (pair[0][0] == pair[1][0]) and pair[0][1] not in self.__non_ligands:
                if pair[0][0] not in self.__id_links:
                    continue
                if pair[0][0] not in self.__id_buffer:
                    self.__id_buffer[pair[0][0]] = 0
                else:
                    self.__id_buffer[pair[0][0]] += 1
                if self.__id_buffer[pair[0][0]] == 0:
                    first = pair[0][0]
                    second = self.__id_links[pair[0][0]][0]
                else:
                    first, second = (
                        self.__id_links[pair[0][0]][self.__id_buffer[pair[0][0]] - 1],
                        self.__id_links[pair[0][0]][self.__id_buffer[pair[0][0]]],
                    )
                if pair[0][1] < pair[1][1]:
                    pair[0] = [first, 1, pair[0][2]]
                    pair[1] = [second, 1, pair[1][2]]
                else:
                    pair[0] = [first, 1, pair[0][2]]
                    pair[1] = [second, 2, pair[1][2]]
            yaml_string += self.add_title("bond")
            yaml_string += self.add_key_and_value("atom1", pair[0])
            yaml_string += self.add_key_and_value("atom2", pair[1])
        return yaml_string

    def add_version_number(self, version: str):
        return f"version: {version}\n"

    def add_non_indented_string(self, string: str):
        return f"{string}:\n"

    def add_id(self, id_: Union[str, list, int]):
        if isinstance(id_, list):
            self.__ids.extend(i for i in id_ if i not in self.__ids)
            new_id = ", ".join(str(i).replace('"', "").replace("'", "") for i in id_)
        else:
            if id_ not in self.__ids:
                self.__ids.append(id_)
            new_id = str(id_).replace('"', "").replace("'", "")
        if isinstance(id_, list):
            return f"{DELIM}{DELIM}id: [{new_id}]\n"
        return f"{DELIM}{DELIM}id: {new_id}\n"

    def add_sequence(self, sequence: str):
        return f"{DELIM}{DELIM}sequence: {sequence}\n"

    def add_msa(self, msa: Union[str, Path]):
        if not Path(msa).exists() and self.__create_files:
            logger.critical(f"File {msa} does not exist")
            raise FileNotFoundError()
        return f"{DELIM}{DELIM}msa: {msa}\n"

    def add_modifications(self, list_of_modifications: list):
        yaml_string = f"{DELIM}{DELIM}modifications:\n"
        for modification in list_of_modifications:
            yaml_string += f"{DELIM}{DELIM}{DELIM}- position: {modification['ptmPosition']}\n"
            yaml_string += f"{DELIM}{DELIM}{DELIM}  ccd: {modification['ptmType']}\n"
        return yaml_string

    def add_key_and_value(self, key: str, value: str):
        if key == "smiles":
            val = value.replace("'", "''")
            return f"{DELIM}{DELIM}{key}: '{val}'\n"
        return f"{DELIM}{DELIM}{key}: {value}\n"

    def add_ligand_information(self, ligand_dict: dict, linked_id=None):
        if "ccdCodes" in ligand_dict and len(ligand_dict["ccdCodes"]) == 0:
            return ""
        yaml_string = self.add_title("ligand")
        yaml_string += self.add_id(ligand_dict["id"])
        if "smiles" in ligand_dict:
            yaml_string += self.add_key_and_value("smiles", ligand_dict["smiles"])
        elif "ccdCodes" in ligand_dict:
            if isinstance(ligand_dict["ccdCodes"], str):
                yaml_string += self.add_key_and_value("ccd", ligand_dict["ccdCodes"])
            elif isinstance(ligand_dict["ccdCodes"], list):
                if linked_id is not None:
                    self.__add_linked_ids(linked_id, ligand_dict["id"])
                yaml_string += self.add_key_and_value("ccd", ligand_dict["ccdCodes"][0])
                yaml_string += self.add_ligand_information(
                    {
                        "id": self.find_next_id(),
                        "ccdCodes": ligand_dict["ccdCodes"][1:],
                    },
                    linked_id=ligand_dict["id"],
                )
        else:
            logger.critical("Ligand must have either a smiles or ccdCodes")
            raise ValueError()
        return yaml_string

    def add_sequence_information(self, sequence_dict: dict):
        yaml_string = self.add_id(sequence_dict["id"])
        if "sequence" in sequence_dict:
            yaml_string += self.add_sequence(sequence_dict["sequence"])
        if isinstance(sequence_dict["id"], str):
            id_ = [sequence_dict["id"]]
        else:
            id_ = sequence_dict["id"]
        self.__non_ligands.extend(id_)
        if self.msa_file is not None:
            if self.__create_files:
                self.msa_to_file(sequence_dict["unpairedMsa"], self.msa_file)
            yaml_string += self.add_msa(self.msa_file)
        if "modifications" in sequence_dict and sequence_dict["modifications"]:
            yaml_string += self.add_modifications(sequence_dict["modifications"])
        return yaml_string

    def add_title(self, name: str):
        return f"{DELIM}- {name}:\n"

    def sequence_to_yaml(self, sequence_dict: dict, yaml_string: str = ""):
        for sequence_type, sequence_info_dict in sequence_dict.items():
            yaml_string += self.add_title(sequence_type)
            self.msa_file = (
                Path(self.working_dir)
                / f"{''.join(random.choices(string.ascii_letters, k=5))}.a3m"
                if "unpairedMsa" in sequence_info_dict
                else None
            )
            yaml_string += self.add_sequence_information(sequence_info_dict)
        return yaml_string

    def write_yaml(self, file_path: Union[str, Path]):
        assert self.yaml_string, "No yaml string to write to file"
        assert Path(file_path).suffix == ".yaml", "File must have a .yaml extension"
        Path(file_path).write_text(self.yaml_string)

    def find_next_id(self):
        if self.__id_char not in self.__ids:
            return self.__id_char
        while self.__id_char in self.__ids:
            self.__id_char = chr(ord(self.__id_char) + 1)
        return self.__id_char

    def get_ids(self, sequences: list):
        for sequence in sequences:
            for key in sequence:
                for key2 in sequence[key]:
                    if key2 == "id":
                        if isinstance(sequence[key][key2], list):
                            self.__ids.extend(sequence[key][key2])
                        else:
                            self.__ids.append(sequence[key][key2])

    def __add_linked_ids(self, ligand_id: Union[str, int], linked_ligand_id: Union[str, int]):
        if not self.__id_links:
            self.__id_links[ligand_id] = [linked_ligand_id]
            return
        for id_, value in self.__id_links.items():
            if ligand_id in value:
                self.__id_links[id_].append(linked_ligand_id)
