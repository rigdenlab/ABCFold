import json
import logging
import random
import string
from pathlib import Path
from typing import Any, Dict, List, Union

logger = logging.getLogger("logger")


ION_CCD_CODES = [
    # alkali / alkaline earth
    "LI", "NA", "K", "RB", "CS",
    "MG", "CA", "SR", "BA",
    # transition metals
    "ZN", "FE", "FE2", "FE3", "CU", "MN", "CO", "NI",
    "CD", "HG",
    # halides
    "F", "CL", "BR", "I",
]


class ProtenixJson:
    """
    Object to convert an AlphaFold3 json file to a Protenix JSON file.
    """

    def __init__(self, working_dir: Union[str, Path], create_files: bool = True):
        self.working_dir = working_dir
        self.seeds: list = [42]
        self.__ids: Dict = {}
        self.__id_counter: int = 1
        self.__create_files = create_files
        self.protenix_dict: Dict = {}

    @property
    def chain_ids(self) -> Dict:
        return self.__ids

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

    def json_to_json(
        self,
        json_file_or_dict: Union[dict, str, Path],
    ):
        """
        Main function to convert an AF3 json file or dict to a Protenix json string

        Args:
            json_file_or_dict (Union[dict, str, Path]): json file or dict

        Returns:
            Dict: protenix dictionary
        """
        logger.info("Converting input json to a Protenix compatible json file")
        if isinstance(json_file_or_dict, str) or isinstance(json_file_or_dict, Path):
            with open(json_file_or_dict, "r") as f:
                json_dict = json.load(f)
        else:
            json_dict = json_file_or_dict

        protenix_sequences = []
        for key, value in json_dict.items():
            if key == "name":
                self.protenix_dict["name"] = value
            if key == "modelSeeds":
                if isinstance(value, list):
                    self.seeds = value
                elif isinstance(value, int):
                    self.seeds = [value]
                self.protenix_dict["modelSeeds"] = self.seeds

            if key == "sequences":
                for entry in value:
                    if "protein" in entry:
                        protein_entry = self.convert_protein(entry["protein"])
                        protenix_sequences.append(protein_entry)
                    elif "rna" in entry:
                        rna_entry = self.convert_rna(entry["rna"])
                        protenix_sequences.append(rna_entry)
                    elif "dna" in entry:
                        dna_entry = self.convert_dna(entry["dna"])
                        protenix_sequences.append(dna_entry)
                    elif "ligand" in entry:
                        ligand_entry = self.convert_ligand(entry["ligand"])
                        protenix_sequences.append(ligand_entry)

            if key == "bondedAtomPairs" and value is not None:
                contact = self.convert_bonded_atom_pairs(value)
                self.protenix_dict["contact"] = contact

        self.protenix_dict["sequences"] = protenix_sequences

        return self.protenix_dict

    def convert_protein(self, seq_dict) -> Dict[str, Any]:
        sequence = seq_dict["sequence"]
        chain_ids = seq_dict.get("id", [])
        for chain in chain_ids:
            self.__ids[chain] = self.__id_counter
            self.__id_counter += 1
        count = len(chain_ids) if chain_ids else 1

        protein_chain = {
            "sequence": sequence,
            "count": count,
        }

        modifications = seq_dict.get("modifications")
        if modifications:
            prefixed = []
            for mod in modifications:
                m = dict(mod)
                typ = str(m.get("ptmType", ""))
                if not typ.startswith("CCD_"):
                    typ = "CCD_" + typ
                m["ptmType"] = typ
                prefixed.append(m)
            protein_chain["modifications"] = prefixed

        unpaired_msa = seq_dict.get("unpairedMsa")
        paired_msa = seq_dict.get("pairedMsa")
        random_string = ''.join(random.choices(string.ascii_letters, k=5))
        msa_dir = Path(self.working_dir) / random_string
        if unpaired_msa and self.__create_files:
            if not msa_dir.exists():
                msa_dir.mkdir(parents=True, exist_ok=True)
            self.msa_to_file(
                unpaired_msa,
                msa_dir / "non_pairing.a3m"
            )

        if paired_msa and self.__create_files:
            if not msa_dir.exists():
                msa_dir.mkdir(parents=True, exist_ok=True)
            self.msa_to_file(
                paired_msa,
                msa_dir / "pairing.a3m"
            )

        if unpaired_msa or paired_msa and self.__create_files:
            protein_chain["msa"] = {
                "precomputed_msa_dir": msa_dir.as_posix(),
                "pairing_db": "uniref100"
            }

        return {
            "proteinChain": protein_chain
        }

    def convert_rna(self, seq_dict) -> Dict[str, Any]:
        sequence = seq_dict["sequence"]
        chain_ids = seq_dict.get("id", [])
        for chain in chain_ids:
            self.__ids[chain] = self.__id_counter
            self.__id_counter += 1
        count = len(chain_ids) if chain_ids else 1

        rna_chain = {
            "sequence": sequence,
            "count": count,
        }

        modifications = seq_dict.get("modifications")
        if modifications:
            prefixed = []
            for mod in modifications:
                m = dict(mod)
                typ = str(m.get("modificationType", ""))
                if not typ.startswith("CCD_"):
                    typ = "CCD_" + typ
                m["modificationType"] = typ
                prefixed.append(m)
            rna_chain["modifications"] = prefixed

        return {
            "rnaSequence": rna_chain
        }

    def convert_dna(self, seq_dict) -> Dict[str, Any]:
        sequence = seq_dict["sequence"]
        chain_ids = seq_dict.get("id", [])
        for chain in chain_ids:
            self.__ids[chain] = self.__id_counter
            self.__id_counter += 1
        count = len(chain_ids) if chain_ids else 1

        dna_chain = {
            "sequence": sequence,
            "count": count,
        }

        modifications = seq_dict.get("modifications")
        if modifications:
            prefixed = []
            for mod in modifications:
                m = dict(mod)
                typ = str(m.get("modificationType", ""))
                if not typ.startswith("CCD_"):
                    typ = "CCD_" + typ
                m["modificationType"] = typ
                prefixed.append(m)
            dna_chain["modifications"] = prefixed

        return {
            "dnaSequence": dna_chain
        }

    def convert_ligand(self, seq_dict) -> Dict[str, Any]:
        chain_ids = seq_dict.get("id", [])
        for chain in chain_ids:
            self.__ids[chain] = self.__id_counter
            self.__id_counter += 1
        count = len(chain_ids) if chain_ids else 1

        if "ccdCodes" in seq_dict:
            ligand_id = seq_dict["ccdCodes"][0]
            ligand = f"CCD_{ligand_id}"
        else:
            ligand = seq_dict["smiles"]
            ligand_id = ligand

        if ligand_id.upper() in ION_CCD_CODES:
            ligand_chain = {
                "ion": ligand,
                "count": count,
            }
            return {
                "ion": ligand_chain
            }
        else:
            ligand_chain = {
                "ligand": ligand,
                "count": count,
            }
            return {
                "ligand": ligand_chain
            }

    def convert_bonded_atom_pairs(self,
                                  bonded_atom_pairs: List[List[List[Union[str, int]]]]):
        contacts = []
        for pair in bonded_atom_pairs:
            entity1, position1, atom1 = pair[0]
            entity2, position2, atom2 = pair[1]
            contact = {
                "entity1": self.__ids[entity1],
                "position1": position1,
                "copy1": 1,
                "atom1": atom1,
                "entity2": self.__ids[entity2],
                "position2": position2,
                "copy2": 1,
                "atom2": atom2,
                "max_distance": 6,
                "min_distance": 0
            }
            contacts.append(contact)
        return contacts

    def write_json(self, out_file: Union[str, Path]):
        """
        Write the Protenix json to a file

        Args:
            out_file (Union[str, Path]): output file path

        Returns:
            None
        """

        with open(out_file, "w") as f:
            json.dump([self.protenix_dict], f, indent=4)
