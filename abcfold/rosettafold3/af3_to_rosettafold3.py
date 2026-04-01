import json
import logging
import random
import string
from pathlib import Path
from typing import Any, Dict, Union

logger = logging.getLogger("logger")


class Rosettafoldjson:
    """
    Object to convert an AlphaFold3 json file to a RosettaFold3 JSON file.
    """

    def __init__(self, working_dir: Union[str, Path],
                 create_files: bool = True):
        self.working_dir = working_dir
        self.seeds: list = [42]
        self.__ids: Dict = {}
        self.__create_files = create_files
        self.name = ""
        self.rosettafold_dict: Dict = {}

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
        Main function to convert an AF3 json file or dict to a RosettaFold3 json string

        Args:
            json_file_or_dict (Union[dict, str, Path]): json file or dict

        Returns:
            Dict: RosettaFold3 dictionary
        """
        logger.info("Converting input json to a RosettaFold3 compatible json file")
        if isinstance(json_file_or_dict, str) or isinstance(json_file_or_dict, Path):
            with open(json_file_or_dict, "r") as f:
                json_dict = json.load(f)
        else:
            json_dict = json_file_or_dict

        rosettafold_sequences = []
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
                    if "protein" in entry:
                        for chain_id in entry["protein"].get("id", []):
                            chain_entry = self.convert_component(entry["protein"],
                                                                 chain_id)
                            rosettafold_sequences.append(chain_entry)
                    elif "rna" in entry:
                        for chain_id in entry["rna"].get("id", []):
                            chain_entry = self.convert_component(entry["rna"],
                                                                 chain_id)
                            rosettafold_sequences.append(chain_entry)
                    elif "dna" in entry:
                        for chain_id in entry["dna"].get("id", []):
                            chain_entry = self.convert_component(entry["dna"],
                                                                 chain_id)
                            rosettafold_sequences.append(chain_entry)
                    elif "ligand" in entry:
                        for chain_id in entry["ligand"].get("id", []):
                            chain_entry = self.convert_ligand(entry["ligand"])
                            rosettafold_sequences.append(chain_entry)

        self.rosettafold_dict = {
            "name": self.name,
            "components": rosettafold_sequences
        }

        return self.rosettafold_dict

    def convert_component(self, seq_dict, chain_id) -> Dict[str, Any]:
        sequence = seq_dict["sequence"]
        modifications = seq_dict.get("modifications", [])
        unpaired_msa = seq_dict.get("unpairedMsa")

        sequence_list = list(sequence)
        if modifications:
            for mod in modifications:
                if "ptmType" in mod and "ptmPosition" in mod:
                    ptm_type = mod['ptmType']
                    position = int(mod['ptmPosition']) - 1
                    sequence_list[position] = f"({ptm_type})"
                    if unpaired_msa is not None:
                        msa_lines = unpaired_msa.splitlines()
                        input_seq = msa_lines[1]
                        idx = int(mod['ptmPosition']) - 1
                        input_seq = input_seq[:idx] + 'X' + input_seq[idx+1:]
                        msa_lines[1] = input_seq
                        seq_dict['unpairedMsa'] = "\n".join(msa_lines)
                elif "modificationType" in mod and "basePosition" in mod:
                    mod_type = mod['modificationType']
                    position = int(mod['basePosition']) - 1
                    sequence_list[position] = f"({mod_type})"
        sequence = ''.join(sequence_list)

        chain = {
            "seq": sequence,
            "chain_id": chain_id,
        }

        random_string = ''.join(random.choices(string.ascii_letters, k=5))
        msa_dir = Path(self.working_dir) / random_string
        if unpaired_msa and self.__create_files:
            msa_out = msa_dir / "colabfold_main.a3m"
            if not msa_dir.exists():
                msa_dir.mkdir(parents=True, exist_ok=True)
            self.msa_to_file(
                unpaired_msa,
                msa_out
            )
            chain["msa_path"] = msa_out.resolve().as_posix()

        return chain

    def convert_ligand(self, seq_dict) -> Dict[str, Any]:
        ligand_chain = {}

        if "ccdCodes" in seq_dict:
            ligand_id = seq_dict["ccdCodes"][0]
            ligand_chain["ccd_code"] = ligand_id
        else:
            ligand_id = seq_dict["smiles"]
            ligand_chain["smiles"] = ligand_id

        return ligand_chain

    def write_json(self, out_file: Union[str, Path]):
        """
        Write the RosettaFold3 json to a file

        Args:
            out_file (Union[str, Path]): output file path

        Returns:
            None
        """

        with open(out_file, "w") as f:
            json.dump([self.rosettafold_dict], f, indent=4)
