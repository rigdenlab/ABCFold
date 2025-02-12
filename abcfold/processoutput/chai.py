import json
import logging
from pathlib import Path
from typing import Union

from abcfold.chai1.af3_to_chai import ChaiFasta
from abcfold.processoutput.file_handlers import (CifFile, ConfidenceJsonFile,
                                                 FileTypes, NpyFile, NpzFile)
from abcfold.processoutput.utils import Af3Pae

logger = logging.getLogger("logger")


class ChaiOutput:
    def __init__(
        self,
        chai_output_dir: Union[str, Path],
        input_params: dict,
        name: str,
    ):
        """
        Object to process the output of an Chai-1 run

        Args:
            chai_output_dir (Union[str, Path]): Path to the Chai-1 output directory
            input_params (dict): Dictionary containing the input parameters used for the
            Chai-1 run
            name (str): Name given to the Chai-1 run

        Attributes:
            input_params (dict): Dictionary containing the input parameters used for the
            Chai-1 run
            output_dir (Path): Path to the Chai-1 output directory
            name (str): Name given to the Chai-1 run
            output (dict): Dictionary containing the processed output the contents
            of the Chai-1 output directory. The dictionary is structured as follows:

            {
                1: {
                    "pae": NpzFile,
                    "cif": CifFile,
                    "scores": NpyFile
                },
                2: {
                    "pae": NpzFile,
                    "cif": CifFile,
                    "scores": NpyFile
                },
                ...
            }
            pae_files (list): Ordered list of NpzFile objects containing the PAE data
            cif_files (list): Ordered list of CifFile objects containing the CIF data
            scores_files (list):  Ordered list of NpyFile objects containing the scores
            data

        """
        self.input_params = input_params
        self.output_dir = Path(chai_output_dir)
        self.name = name

        if not self.output_dir.name.startswith("chai1_" + self.name):
            self.output_dir = self.output_dir.rename(
                self.output_dir.parent / f"chai1_{self.name}"
            )
        self.input_fasta = self.get_input_fasta()

        self.output = self.process_chai_output()
        self.pae_files = [
            value["pae"] for value in self.output.values() if "pae" in value
        ]
        self.cif_files = [
            value["cif"] for value in self.output.values() if "cif" in value
        ]
        self.pae_to_af3()
        self.scores_files = [
            value["scores"] for value in self.output.values() if "scores" in value
        ]
        self.af3_pae_files = [
            value["af3_pae"] for value in self.output.values() if "af3_pae" in value
        ]

    def process_chai_output(self):
        file_groups = {}

        for pathway in self.output_dir.iterdir():
            number = pathway.stem.split("model_idx_")[-1]
            if number.isdigit():
                number = int(number)

            file_type = pathway.suffix[1:]

            if file_type == FileTypes.NPZ.value:
                file_ = NpzFile(str(pathway))

            elif file_type == FileTypes.CIF.value:
                file_ = CifFile(str(pathway), self.input_params)
                file_ = self.update_chain_labels(file_)

            elif file_type == FileTypes.NPY.value:
                file_ = NpyFile(str(pathway))
            else:
                continue

            if isinstance(number, str):
                number = -1

            if number not in file_groups:
                file_groups[number] = [file_]
            else:
                file_groups[number].append(file_)

        model_number_file_type_file = {}
        for model_number, files in file_groups.items():
            intermediate_dict = {}
            for file_ in sorted(files, key=lambda x: x.suffix):
                if file_.pathway.stem.startswith("scores.model"):
                    intermediate_dict["scores"] = file_
                elif file_.pathway.stem.startswith("pred.model"):
                    file_.name = f"Chai-1_{model_number}"
                    # Chai cif not recognised by pae-viewer, so we load and save
                    file_.to_file(file_.pathway)
                    intermediate_dict["cif"] = file_
                elif file_.pathway.stem.startswith("pae_scores"):
                    intermediate_dict["pae"] = file_

            model_number_file_type_file[model_number] = intermediate_dict

        model_number_file_type_file = {
            model_number: model_number_file_type_file[model_number]
            for model_number in sorted(model_number_file_type_file)
        }

        return model_number_file_type_file

    def pae_to_af3(self) -> None:
        """
        Convert the Chai-1 PAE data to the format expected by AlphaFold3

        """

        pae_file = self.pae_files[-1]
        for i, cif_file in enumerate(self.cif_files):
            pae = Af3Pae.from_chai1(
                pae_file.data[i],
                cif_file,
            )

            out_name = self.output_dir.joinpath(cif_file.pathway.stem + "_af3_pae.json")
            with open(out_name, "w") as f:
                json.dump(pae.scores, f)

            self.output[i]["af3_pae"] = ConfidenceJsonFile(out_name)

    def get_input_fasta(self) -> ChaiFasta:
        """
        Function to get the input fasta file used for the Chai-1 run

        Returns:
            ChaiFasta: ChaiFasta object containing the input fasta file

        """

        ch = ChaiFasta(self.output_dir, create_files=False)
        ch.json_to_fasta(self.input_params)

        return ch

    def update_chain_labels(self, cif_file: CifFile) -> CifFile:
        """
        Function to update the chain labels in the CIF file

        Args:
            cif_file (CifFile): CifFile object to update the chain labels for

        """

        cif_file.relabel_chains(self.input_fasta.chain_ids)
        cif_file.to_file(cif_file.pathway)
        return CifFile(cif_file.pathway, self.input_params)
