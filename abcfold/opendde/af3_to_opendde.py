import json
import logging
import random
import string
from pathlib import Path
from typing import Any, Dict, List, Union

logger = logging.getLogger("logger")

ENTITY_KEYS = {
    "protein": "proteinChain",
    "dna": "dnaSequence",
    "rna": "rnaSequence",
}


class OpenDDEjson:
    """
    Object to convert an AlphaFold3 json file to an OpenDDE JSON file.

    OpenDDE reads a list of jobs using AlphaFold-Server style entity keys
    (``proteinChain``, ``dnaSequence``, ``rnaSequence``, ``ligand``) rather than
    the single job ``alphafold3`` dialect.
    """

    def __init__(self, working_dir: Union[str, Path], create_files: bool = True):
        self.working_dir = working_dir
        self.seeds: list = [42]
        self.__create_files = create_files
        self.name = ""
        self.opendde_dict: Dict = {}
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
        Main function to convert an AF3 json file or dict to an OpenDDE dictionary

        Args:
            json_file_or_dict (Union[dict, str, Path]): json file or dict

        Returns:
            Dict: OpenDDE dictionary
        """
        logger.info("Converting input json to an OpenDDE compatible json file")
        if isinstance(json_file_or_dict, (str, Path)):
            with open(json_file_or_dict, "r") as f:
                json_dict = json.load(f)
        else:
            json_dict = json_file_or_dict

        opendde_sequences: List[Dict[str, Any]] = []
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
                    for af3_key, opendde_key in ENTITY_KEYS.items():
                        if af3_key in entry:
                            opendde_sequences.append(
                                {
                                    opendde_key: self.convert_component(
                                        entry[af3_key], af3_key
                                    )
                                }
                            )
                    if "ligand" in entry:
                        opendde_sequences.append(
                            {"ligand": self.convert_ligand(entry["ligand"])}
                        )

        self.opendde_dict = {
            "name": self.name,
            "modelSeeds": self.seeds,
            "sequences": opendde_sequences,
        }

        return self.opendde_dict

    def convert_component(self, seq_dict, entity_type) -> Dict[str, Any]:
        chain_ids = seq_dict.get("id", [])
        if isinstance(chain_ids, str):
            chain_ids = [chain_ids]

        chain: Dict[str, Any] = {
            "sequence": seq_dict["sequence"],
            "count": len(chain_ids) if chain_ids else 1,
        }
        if chain_ids:
            chain["id"] = chain_ids

        modifications = []
        for mod in seq_dict.get("modifications", []):
            if "ptmType" in mod and "ptmPosition" in mod:
                modifications.append(
                    {
                        "ptmType": self.ccd_prefixed(mod["ptmType"]),
                        "ptmPosition": int(mod["ptmPosition"]),
                    }
                )
            elif "modificationType" in mod and "basePosition" in mod:
                modifications.append(
                    {
                        "modificationType": self.ccd_prefixed(
                            mod["modificationType"]
                        ),
                        "basePosition": int(mod["basePosition"]),
                    }
                )
        if modifications:
            chain["modifications"] = modifications

        # OpenDDE takes MSAs as file paths, so any inline AF3 alignment is
        # written out next to the converted input.
        if entity_type in ("protein", "rna"):
            unpaired_msa = seq_dict.get("unpairedMsa")
            if unpaired_msa and self.__create_files:
                random_string = "".join(random.choices(string.ascii_letters, k=5))
                msa_dir = Path(self.working_dir) / random_string
                msa_dir.mkdir(parents=True, exist_ok=True)
                msa_out = msa_dir / "unpaired.a3m"
                self.msa_to_file(unpaired_msa, msa_out)
                chain["unpairedMsaPath"] = msa_out.resolve().as_posix()
                self.uses_msa = True

            paired_msa = seq_dict.get("pairedMsa")
            if paired_msa and self.__create_files and entity_type == "protein":
                random_string = "".join(random.choices(string.ascii_letters, k=5))
                msa_dir = Path(self.working_dir) / random_string
                msa_dir.mkdir(parents=True, exist_ok=True)
                msa_out = msa_dir / "paired.a3m"
                self.msa_to_file(paired_msa, msa_out)
                chain["pairedMsaPath"] = msa_out.resolve().as_posix()
                self.uses_msa = True

        return chain

    @staticmethod
    def ccd_prefixed(code: str) -> str:
        """OpenDDE expects modification and CCD ligand codes prefixed with CCD_"""
        return code if code.startswith("CCD_") else f"CCD_{code}"

    def convert_ligand(self, seq_dict) -> Dict[str, Any]:
        chain_ids = seq_dict.get("id", [])
        if isinstance(chain_ids, str):
            chain_ids = [chain_ids]

        if "ccdCodes" in seq_dict and seq_dict["ccdCodes"]:
            codes = seq_dict["ccdCodes"]
            if isinstance(codes, str):
                codes = [codes]
            # Several CCD codes form one ligand chain, joined by underscores.
            ligand = f"CCD_{'_'.join(codes)}"
        else:
            ligand = seq_dict["smiles"]

        ligand_chain: Dict[str, Any] = {
            "ligand": ligand,
            "count": len(chain_ids) if chain_ids else 1,
        }
        if chain_ids:
            ligand_chain["id"] = chain_ids

        return ligand_chain

    def write_json(self, out_file: Union[str, Path]):
        """
        Write the OpenDDE json to a file

        OpenDDE always reads a list of jobs, even for a single complex.

        Args:
            out_file (Union[str, Path]): output file path

        Returns:
            None
        """

        with open(out_file, "w") as f:
            json.dump([self.opendde_dict], f, indent=4)
