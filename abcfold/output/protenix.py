import logging
from pathlib import Path
from typing import Dict, Union

from abcfold.output.file_handlers import (CifFile, ConfidenceJsonFile,
                                          FileTypes, NpzFile)
from abcfold.output.utils import Af3Pae
from abcfold.protenix.af3_to_protenix import ProtenixJson

logger = logging.getLogger("logger")

FileDict = Dict[str, Union[CifFile, ConfidenceJsonFile]]


class ProtenixOutput:
    def __init__(
        self,
        protenix_output_dirs: list[Union[str, Path]],
        input_params: dict,
        name: str,
        save_input: bool = False,
    ):
        """
        Object to process the output of an Protenix run

        Args:
            protenix_output_dirs (list[Union[str, Path]]): Path to the Protenix
            output directory
            input_params (dict): Dictionary containing the input parameters used for the
            Protenix run
            name (str): Name given to the Protenix run
            save_input (bool): If True, Protenix was run with the save_input flag

        Attributes:
            output_dirs (list): List of paths to the Protenix output directory(s)
            input_params (dict): Dictionary containing the input parameters used for the
            Protenix run
            name (str): Name given to the Protenix run
            output (dict): Dictionary containing the processed output the contents
            of the Protenix output directory(s). The dictionary is structured as
            follows:

            {
                "seed-1": {
                    1: {
                        "cif": CifFile,
                        "scores": ConfidenceJsonFile,
                        "af3_pae": ConfidenceJsonFile,
                    },
                    2: {
                        "cif": CifFile,
                        "scores": ConfidenceJsonFile,
                        "af3_pae": ConfidenceJsonFile,
                    },
                },
                etc...
            }
            pae_files (list): Ordered list of ConfidenceJsonFile objects containing the
            PAE data
            cif_files (list): Ordered list of CifFile objects containing the model data
            scores_files (list): Ordered list of ConfidenceJsonFile objects containing
            the model scores
        """
        self.output_dirs = [Path(x) for x in protenix_output_dirs]
        self.input_params = input_params
        self.name = name
        self.save_input = save_input

        parent_dir = self.output_dirs[0].parent
        new_parent = parent_dir / f"protenix_{self.name}"
        new_parent.mkdir(parents=True, exist_ok=True)

        if self.save_input:
            protenix_json = list(parent_dir.glob("*.json"))[0]
            if protenix_json.exists():  # check this works
                protenix_json.rename(new_parent / "protenix_input.json")

            protenix_msas = list(parent_dir.glob("*/*.a3m"))
            if protenix_msas:
                for protenix_msa in protenix_msas:
                    if protenix_msa.exists():
                        protenix_msa.rename(new_parent / protenix_msa.name)

        new_output_dirs = []
        for output_dir in self.output_dirs:
            if output_dir.name.startswith("protenix_results_"):
                new_path = new_parent / output_dir.name
                output_dir.rename(new_path)
                new_output_dirs.append(new_path)
            else:
                new_output_dirs.append(output_dir)
        self.output_dirs = new_output_dirs

        self.json_input_obj = self.get_input_json()

        self.output = self.process_protenix_output()

        self.seeds = list(self.output.keys())
        self.pae_files = {
            seed: [value["pae"] for value in self.output[seed].values()]
            for seed in self.seeds
        }
        self.cif_files = {
            seed: [value["cif"] for value in self.output[seed].values()]
            for seed in self.seeds
        }
        self.scores_files = {
            seed: [value["score"] for value in self.output[seed].values()]
            for seed in self.seeds
        }
        self.pae_to_af3()
        self.af3_pae_files = {
            seed: [value["af3_pae"] for value in self.output[seed].values()]
            for seed in self.seeds
        }

    def process_protenix_output(self) -> dict:
        """
        Function to process the output of a Protenix run
        """
        file_groups: Dict[str, Dict[int, list]] = {}
        for pathway in self.output_dirs:
            seed = pathway.name.split("_")[-1]
            if seed not in file_groups:
                file_groups[seed] = {}

            for output in pathway.rglob("*"):
                number_str = output.stem.split("_sample_")[-1]
                if not number_str.isdigit():
                    continue
                number_int = int(number_str)

                file_type = output.suffix[1:]

                file_: Union[NpzFile, CifFile, ConfidenceJsonFile]
                if file_type == FileTypes.NPZ.value:
                    file_ = NpzFile(str(output))
                elif file_type == FileTypes.CIF.value:
                    file_ = CifFile(str(output), self.input_params)
                elif file_type == FileTypes.JSON.value:
                    file_ = ConfidenceJsonFile(str(output))
                else:
                    continue
                if number_int not in file_groups[seed]:
                    file_groups[seed][number_int] = [file_]
                else:
                    file_groups[seed][number_int].append(file_)

        seed_dict: Dict[str, Dict[int, FileDict]] = {}
        for seed, models in file_groups.items():
            model_number_file_type_file: Dict[int, FileDict] = {}
            for model_number, files in models.items():
                intermediate_dict: FileDict = {}
                for file_ in sorted(files, key=lambda x: x.suffix):
                    if isinstance(file_, CifFile):
                        if file_.pathway.suffix == ".cif":
                            file_.name = f"protenix_{seed}_{model_number}"
                            file_ = self.update_chain_labels(file_)
                            intermediate_dict["cif"] = file_
                        elif "full_data" in file_.pathway.stem:
                            intermediate_dict["pae"] = file_
                        elif "summary_confidence" in file_.pathway.stem:
                            intermediate_dict["score"] = file_
                    elif isinstance(file_, ConfidenceJsonFile):
                        if "full_data" in file_.pathway.stem:
                            intermediate_dict["pae"] = file_
                        elif "summary_confidence" in file_.pathway.stem:
                            intermediate_dict["score"] = file_

                model_number_file_type_file[model_number] = intermediate_dict

            model_number_file_type_file = {
                key: model_number_file_type_file[key]
                for key in sorted(model_number_file_type_file)
            }
            seed_dict[seed] = model_number_file_type_file

        return seed_dict

    def pae_to_af3(self) -> None:
        """
        Convert the PAE data from Protenix to the format used by Alphafold3

        Returns:
            None
        """
        new_pae_files: Dict[str, list] = {}
        for seed in self.seeds:
            for (pae_file, cif_file) in zip(
                self.pae_files[seed], self.cif_files[seed]
            ):
                pae = Af3Pae.from_protenix(
                    pae_file.data,
                    cif_file,
                )

                out_name = pae_file.pathway

                pae.to_file(out_name)

                if seed not in new_pae_files:
                    new_pae_files[seed] = []
                new_pae_files[seed].append(ConfidenceJsonFile(out_name))

        self.output = {
            seed: {
                i: {
                    "cif": cif_file,
                    "af3_pae": new_pae_files[seed][i],
                    "scores": self.output[seed][i]["score"],
                }
                for i, cif_file in enumerate(self.cif_files[seed])
            }
            for seed in self.seeds
        }

    def update_chain_labels(self, cif_file: CifFile) -> CifFile:
        """
        Function to update the chain labels in the CIF file

        Args:
            cif_file (CifFile): CifFile object to update the chain labels for

        """

        cif_file.relabel_chains(
            list(self.json_input_obj.chain_ids.keys())
        )
        return cif_file

    def get_input_json(self) -> ProtenixJson:
        """
        Get the input json file used for the Protenix run

        Returns:
            ProtenixJson: Object containing the input json file
        """

        pj = ProtenixJson(self.output_dirs[0], create_files=False)
        pj.json_to_json(self.input_params)

        return pj
