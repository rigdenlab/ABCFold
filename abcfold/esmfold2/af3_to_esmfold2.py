import json
import logging
import random
import string
from pathlib import Path
from typing import Any, Dict, List, Union

logger = logging.getLogger("logger")

# ESMFold2 has no command line interface and no input file format: a complex is
# built in Python as a StructurePredictionInput. This writes the intermediate
# description that abcfold's generated runner turns into that object.
POLYMER_TYPES = {"protein": "protein", "rna": "rna", "dna": "dna"}


class Esmfold2json:
    """
    Object to convert an AlphaFold3 json file to an ESMFold2 input description.
    """

    def __init__(self, working_dir: Union[str, Path], create_files: bool = True):
        self.working_dir = working_dir
        self.seeds: list = [42]
        self.__create_files = create_files
        self.name = ""
        self.esmfold2_dict: Dict = {}
        self.uses_msa = False

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
        Main function to convert an AF3 json file or dict to an ESMFold2 dict

        Args:
            json_file_or_dict (Union[dict, str, Path]): json file or dict

        Returns:
            Dict: ESMFold2 dictionary
        """
        logger.info("Converting input json to an ESMFold2 input description")
        if isinstance(json_file_or_dict, (str, Path)):
            with open(json_file_or_dict, "r") as f:
                json_dict = json.load(f)
        else:
            json_dict = json_file_or_dict

        sequences: List[Dict[str, Any]] = []
        for key, value in json_dict.items():
            if key == "name":
                self.name = value
            if key == "modelSeeds":
                if isinstance(value, list):
                    self.seeds = value
                elif isinstance(value, int):
                    self.seeds = [value]
            if key == "sequences":
                for entry in value:
                    for af3_key, entity_type in POLYMER_TYPES.items():
                        if af3_key in entry:
                            sequences.append(
                                self.convert_component(entry[af3_key], entity_type)
                            )
                    if "ligand" in entry:
                        sequences.append(self.convert_ligand(entry["ligand"]))

        self.esmfold2_dict = {
            "name": self.name,
            "seeds": self.seeds,
            "sequences": sequences,
        }

        return self.esmfold2_dict

    def convert_component(self, seq_dict, entity_type) -> Dict[str, Any]:
        chain_ids = seq_dict.get("id", [])
        if isinstance(chain_ids, str):
            chain_ids = [chain_ids]

        chain: Dict[str, Any] = {
            "type": entity_type,
            "id": chain_ids,
            "sequence": seq_dict["sequence"],
        }

        modifications = []
        for mod in seq_dict.get("modifications", []):
            # AF3 positions are 1-based, ESMFold2 Modification.position is
            # zero-indexed.
            if "ptmType" in mod and "ptmPosition" in mod:
                modifications.append(
                    {
                        "position": int(mod["ptmPosition"]) - 1,
                        "ccd": mod["ptmType"],
                    }
                )
            elif "modificationType" in mod and "basePosition" in mod:
                modifications.append(
                    {
                        "position": int(mod["basePosition"]) - 1,
                        "ccd": mod["modificationType"],
                    }
                )
        if modifications:
            chain["modifications"] = modifications

        # DNAInput carries no alignment.
        if entity_type in ("protein", "rna"):
            unpaired_msa = seq_dict.get("unpairedMsa")
            if unpaired_msa and self.__create_files:
                random_string = "".join(random.choices(string.ascii_letters, k=5))
                msa_dir = Path(self.working_dir) / random_string
                msa_dir.mkdir(parents=True, exist_ok=True)
                msa_out = msa_dir / "unpaired.a3m"
                self.msa_to_file(unpaired_msa, msa_out)
                chain["msa_path"] = msa_out.resolve().as_posix()
                self.uses_msa = True

        return chain

    def convert_ligand(self, seq_dict) -> Dict[str, Any]:
        chain_ids = seq_dict.get("id", [])
        if isinstance(chain_ids, str):
            chain_ids = [chain_ids]

        ligand: Dict[str, Any] = {"type": "ligand", "id": chain_ids}

        if "ccdCodes" in seq_dict and seq_dict["ccdCodes"]:
            codes = seq_dict["ccdCodes"]
            if isinstance(codes, str):
                codes = [codes]
            # LigandInput.ccd rejects a bare string and requires a list.
            ligand["ccd"] = list(codes)
        else:
            ligand["smiles"] = seq_dict["smiles"]

        return ligand

    def write_json(self, out_file: Union[str, Path]):
        """
        Write the ESMFold2 input description to a file

        Args:
            out_file (Union[str, Path]): output file path

        Returns:
            None
        """

        if not self.__create_files:
            return

        with open(out_file, "w") as f:
            json.dump(self.esmfold2_dict, f, indent=4)
