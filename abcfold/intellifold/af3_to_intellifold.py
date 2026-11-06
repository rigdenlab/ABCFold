import json
import logging
from pathlib import Path
from typing import Dict, Union

logger = logging.getLogger("logger")

# IntelliFold v2 runs a vendored AlphaFold 3 JAX engine, so it reads the AF3
# dialect directly. Version 2 of the dialect is the lowest that accepts the
# *Path fields used for templates and alignments.
AF3_DIALECT = "alphafold3"
AF3_VERSION = 2


class Intellifoldjson:
    """
    Object to convert an AlphaFold3 json file to an IntelliFold JSON file.

    IntelliFold v2 consumes the AlphaFold 3 dialect, so this normalises the
    input rather than remapping it: seeds are collected, and any chain carrying
    an unpaired alignment gets an explicit empty ``pairedMsa`` so the engine
    does not attempt its own pairing.
    """

    def __init__(self, working_dir: Union[str, Path], create_files: bool = True):
        self.working_dir = working_dir
        self.seeds: list = [42]
        self.__create_files = create_files
        self.name = ""
        self.intellifold_dict: Dict = {}
        self.uses_msa = False

    def json_to_json(
        self,
        json_file_or_dict: Union[dict, str, Path],
    ):
        """
        Main function to convert an AF3 json file or dict to an IntelliFold dict

        Args:
            json_file_or_dict (Union[dict, str, Path]): json file or dict

        Returns:
            Dict: IntelliFold dictionary
        """
        logger.info("Converting input json to an IntelliFold compatible json file")
        if isinstance(json_file_or_dict, (str, Path)):
            with open(json_file_or_dict, "r") as f:
                json_dict = json.load(f)
        else:
            json_dict = json_file_or_dict

        intellifold_dict = json.loads(json.dumps(json_dict))

        self.name = intellifold_dict.get("name", "")

        seeds = intellifold_dict.get("modelSeeds", [42])
        if isinstance(seeds, int):
            seeds = [seeds]
        if not seeds:
            seeds = [42]
        self.seeds = seeds
        intellifold_dict["modelSeeds"] = seeds

        for entry in intellifold_dict.get("sequences", []):
            for key in ("protein", "rna"):
                if key not in entry:
                    continue
                chain = entry[key]
                if chain.get("unpairedMsa") or chain.get("unpairedMsaPath"):
                    self.uses_msa = True
                    # An explicit empty pairedMsa is required whenever
                    # unpairedMsa is set by hand.
                    if key == "protein" and not chain.get("pairedMsaPath"):
                        chain["pairedMsa"] = chain.get("pairedMsa", "")
            if "protein" in entry:
                entry["protein"].setdefault("templates", [])

        intellifold_dict["dialect"] = AF3_DIALECT
        intellifold_dict["version"] = max(
            int(intellifold_dict.get("version", AF3_VERSION)), AF3_VERSION
        )

        self.intellifold_dict = intellifold_dict

        return self.intellifold_dict

    def write_json(self, out_file: Union[str, Path]):
        """
        Write the IntelliFold json to a file

        Args:
            out_file (Union[str, Path]): output file path

        Returns:
            None
        """

        if not self.__create_files:
            return

        with open(out_file, "w") as f:
            json.dump(self.intellifold_dict, f, indent=4)
