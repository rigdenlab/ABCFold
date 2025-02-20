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
    Object to convert an AlphaFold3 json file to a boltzmann yaml file.
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
        """
        Takes a msa string and writes it to a file

        Args:
            msa (str): msa string
            file_path (Union[str, Path]): file path to write the msa to

        Returns:
            None
        """

        with open(file_path, "w") as f:
            f.write(msa)

    def json_to_yaml(
        self,
        json_file_or_dict: Union[dict, str, Path],
    ):
        """
        Main function to convert a json file or dict to a yaml string

        Args:
            json_file_or_dict (Union[dict, str, Path]): json file or dict

        Returns:
            str: a string representation of string
        """
        if isinstance(json_file_or_dict, str) or isinstance(json_file_or_dict, Path):
            with open(json_file_or_dict, "r") as f:
                json_dict = json.load(f)
        else:
            json_dict = json_file_or_dict

        self.get_ids(json_dict["sequences"])

        self.yaml_string = ""
        bonded_atom_string = ""

        self.yaml_string += self.add_version_number("1")
        for key, value in json_dict.items():
            if key == "sequences":
                if "sequences" not in self.yaml_string:
                    self.yaml_string += self.add_non_indented_string("sequences")
                for sequence_dict in value:
                    if any([key in sequence_dict for key in ["protein", "rna", "dna"]]):
                        self.yaml_string += self.sequence_to_yaml(sequence_dict)
                    if any([key in sequence_dict for key in ["ligand"]]):
                        self.yaml_string += self.add_ligand_information(
                            sequence_dict["ligand"]
                        )
            if key == "bondedAtomPairs" and isinstance(value, list):

                bonded_atom_string += self.bonded_atom_pairs_to_yaml(value)
                if "constraints" not in self.yaml_string and bonded_atom_string:
                    self.yaml_string += self.add_non_indented_string("constraints")
                self.yaml_string += bonded_atom_string

        return self.yaml_string

    def bonded_atom_pairs_to_yaml(self, bonded_atom_pairs: list):
        yaml_string = ""
        # counter = 0
        for pair in bonded_atom_pairs:

            if (pair[0][0] == pair[1][0]) and pair[0][1] not in self.__non_ligands:

                if pair[0][0] not in self.__id_links:
                    continue

                # I'm sorry
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
        """
        Adds the version number to the yaml string

        Args:
            version (str): version number

        Returns:
            str: yaml string
        """
        return f"version: {version}\n"

    def add_non_indented_string(self, string: str):
        """
        Adds the sequence string to the yaml string

        Returns:
            str: yaml string
        """
        return f"{string}:\n"

    def add_id(self, id_: Union[str, list, int]):
        """
        Adds the id to the yaml string

        Args:
            id_ (Union[str, list, int]): id

        Returns:
            str: yaml string

        """

        if isinstance(id_, list):
            self.__ids.extend([id__ for id__ in id_ if id__ not in self.__ids])
            new_id = ", ".join([str(i).replace('"', "").replace("'", "") for i in id_])
        else:
            self.__ids.append(id_) if id_ not in self.__ids else None
            new_id = str(id_).replace('"', "").replace("'", "")

        return (
            f"{DELIM}{DELIM}id: {new_id}\n"
            if not isinstance(id_, list)
            else f"{DELIM}{DELIM}id: [{new_id}]\n"
        )

    def add_sequence(self, sequence: str):
        """
        Adds the sequence to the yaml string

        Args:
            sequence (str): sequence

        Returns:
            str: yaml string

        """
        return f"{DELIM}{DELIM}sequence: {sequence}\n"

    def add_msa(self, msa: Union[str, Path]):
        """
        Adds the msa file_path to the yaml string, double tabbed

        Args:
            msa (str): msa file_path

        Returns:
            str: yaml string
        """
        if not Path(msa).exists() and self.__create_files:
            msg = f"File {msa} does not exist"
            logger.critical(msg)
            raise FileNotFoundError()
        return f"{DELIM}{DELIM}msa: {msa}\n"

    def add_modifications(self, list_of_modifications: list):
        """
        Adds the modifications to the yaml string, double tabbed

        Args:
            list_of_modifications (list): list of modifications

        Returns:
            str: yaml string
        """
        yaml_string = ""
        yaml_string += f"{DELIM}{DELIM}modifications:\n"
        for modification in list_of_modifications:
            yaml_string += (
                f"{DELIM}{DELIM}{DELIM}- position: {modification['ptmPosition']}\n"
            )
            yaml_string += f"{DELIM}{DELIM}{DELIM}  ccd: {modification['ptmType']}\n"
        return yaml_string

    def add_key_and_value(self, key: str, value: str):
        """
        Adds the key and value to the yaml string, double tabbed

        Args:
            key (str): The key on the left of ':'
            value (str): The value on the right of ':'
        Returns:
            str: yaml string
        """
        return f"{DELIM}{DELIM}{key}: {value}\n"

    def add_ligand_information(self, ligand_dict: dict, linked_id=None):
        """
        Function to add ligand information to the yaml string

        Args:
            ligand_dict (dict): ligand dict

        Returns:
            str: yaml string
        """

        if "ccdCodes" in ligand_dict and len(ligand_dict["ccdCodes"]) == 0:
            return ""
        yaml_string = ""
        yaml_string += self.add_title("ligand")
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

            msg = "Ligand must have either a smiles or ccdCCodes"
            logger.critical(msg)
            raise ValueError()

        return yaml_string

    def add_sequence_information(self, sequence_dict: dict):
        """
        Adds the sequence information of protein, rna, dna to the yaml string

        Args:
            sequence_dict (dict): sequence dict
            msa_file (Union[str, Path]): msa file_path

        Returns:
            str: yaml string
        """
        yaml_string = ""
        yaml_string += self.add_id(sequence_dict["id"])
        yaml_string += (
            self.add_sequence(sequence_dict["sequence"])
            if "sequence" in sequence_dict
            else ""
        )
        if isinstance(sequence_dict["id"], str):
            id_ = [sequence_dict["id"]]
        else:
            id_ = sequence_dict["id"]

        self.__non_ligands.extend(id_)

        if self.msa_file is not None:
            (
                self.msa_to_file(sequence_dict["unpairedMsa"], self.msa_file)
                if self.__create_files
                else None
            )
            yaml_string += self.add_msa(self.msa_file)

        if "modifications" in sequence_dict and sequence_dict["modifications"]:
            yaml_string += self.add_modifications(sequence_dict["modifications"])

        return yaml_string

    def add_title(self, name: str):
        """
        Adds the title to the yaml string

        args:
            name (str): name of the title

        Returns:
            str: yaml string
        """
        return f"{DELIM}- {name}:\n"

    def sequence_to_yaml(self, sequence_dict: dict, yaml_string: str = ""):
        """
        Adds the sequence information to the yaml string

        Args:
            sequence_dict (dict): sequence dict
            yaml_string (str): yaml string

        Returns:
            str: yaml string
        """
        for sequence_type, sequence_info_dict in sequence_dict.items():
            yaml_string += self.add_title(sequence_type)
            self.msa_file = (
                (
                    Path(self.working_dir)
                    / f"{''.join(random.choices(string.ascii_letters, k=5))}.a3m"
                )
                if "unpairedMsa" in sequence_info_dict
                else None
            )

            yaml_string += self.add_sequence_information(sequence_info_dict)

        return yaml_string

    def write_yaml(self, file_path: Union[str, Path]):
        """
        Writes the yaml string to a file

        Args:
            file_path (Union[str, Path]): file path

        Returns:
            None
        """

        assert self.yaml_string, "No yaml string to write to file"
        assert Path(file_path).suffix == ".yaml", "File must have a .yaml extension"
        with open(file_path, "w") as f:
            f.write(self.yaml_string)

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
                            continue
                        self.__ids.append(sequence[key][key2])

    def __add_linked_ids(
        self, ligand_id: Union[str, int], linked_ligand_id: Union[str, int]
    ):
        if not self.__id_links:
            self.__id_links[ligand_id] = [linked_ligand_id]
            return
        for id_, value in self.__id_links.items():
            if ligand_id in value:
                self.__id_links[id_].append(linked_ligand_id)

                return
