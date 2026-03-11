import logging
import shutil
from pathlib import Path
from typing import Union

from abcfold.chai1.af3_to_chai import ChaiFasta
from abcfold.output.file_handlers import (CifFile, ConfidenceJsonFile,
                                          FileTypes, NpyFile, NpzFile)
from abcfold.output.utils import Af3Pae

logger = logging.getLogger("logger")


class ChaiOutput:
    def __init__(
        self,
        chai_output_dirs: list[Union[str, Path]],
        input_params: dict,
        name: str,
        config: dict,
        save_input: bool = False,
    ):
        """
        Object to process the output of an Chai-1 run

        Args:
            chai_output_dirs (list[Union[str, Path]]): Path to the Chai-1
            output directory
            input_params (dict): Dictionary containing the input parameters used for the
            Chai-1 run
            name (str): Name given to the Chai-1 run
            config (dict): Configuration dictionary
            save_input (bool): If True, Chai-1 was run with the save_input flag

        Attributes:
            input_params (dict): Dictionary containing the input parameters used for the
            Chai-1 run
            output_dirs (Path):  List of paths to the Chai-1 output directory(s)
            name (str): Name given to the Chai-1 run
            output (dict): Dictionary containing the processed output the contents
            of the Chai-1 output directory(s). The dictionary is structured as follows:

            {
                "seed-1": {
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
                etc...
            }
            pae_files (list): Ordered list of NpzFile objects containing the PAE data
            cif_files (list): Ordered list of CifFile objects containing the CIF data
            scores_files (list):  Ordered list of NpyFile objects containing the scores
            data

        """
        self.input_params = input_params
        self.output_dirs = [Path(x) for x in chai_output_dirs]
        self.name = name
        self.config = config
        self.save_input = save_input

        parent_dir = self.output_dirs[0].parent
        new_parent = parent_dir / f"chai1_{self.name}"
        new_parent.mkdir(parents=True, exist_ok=True)

        if self.save_input:
            chai_fasta = parent_dir / "chai1.fasta"
            if chai_fasta.exists():
                chai_fasta.rename(new_parent / "chai1.fasta")
            chai_msas = list(parent_dir.glob("*.aligned.pqt"))
            if chai_msas:
                for chai_msa in chai_msas:
                    if chai_msa.exists():
                        chai_msa.rename(new_parent / chai_msa.name)

        new_output_dirs = []
        for output_dir in self.output_dirs:
            if output_dir.name.startswith("chai_output_"):
                new_path = new_parent / output_dir.name
                output_dir.rename(new_path)
                new_output_dirs.append(new_path)
            else:
                new_output_dirs.append(output_dir)
        self.output_dirs = new_output_dirs

        self.input_fasta = self.get_input_fasta()

        self.output = self.process_chai_output()
        self.seeds = list(self.output.keys())

        self.pae_files = {
            seed: [value["pae"] for value in self.output[seed].values()
                   if "pae" in value] for seed in self.seeds
        }
        self.cif_files = {
            seed: [value["cif"] for value in self.output[seed].values()]
            for seed in self.seeds
        }
        self.scores_files = {
            seed: [value["scores"] for value in self.output[seed].values()]
            for seed in self.seeds
        }
        self.pae_to_af3()
        self.af3_pae_files = {
            seed: [value["af3_pae"] for value in self.output[seed].values()]
            for seed in self.seeds
        }

    def process_chai_output(self):
        file_groups: dict[str, dict[int, list]] = {}

        for pathway in self.output_dirs:
            seed = pathway.name.split("_")[-1]
            if seed not in file_groups:
                file_groups[seed] = {}

            for output in pathway.rglob("*"):
                number_str = output.stem.split("model_idx_")[-1]
                if number_str.isdigit():
                    number = int(number_str)

                file_type = output.suffix[1:]

                file_: Union[NpzFile, CifFile, NpyFile]
                if file_type == FileTypes.NPZ.value:
                    file_ = NpzFile(str(output))

                elif file_type == FileTypes.CIF.value:
                    file_ = CifFile(str(output), self.input_params)
                    file_ = self.update_chain_labels(file_)

                elif file_type == FileTypes.NPY.value:
                    file_ = NpyFile(str(output))
                else:
                    continue

                if isinstance(number, str):
                    number = -1

                if number not in file_groups[seed]:
                    file_groups[seed][number] = [file_]
                else:
                    file_groups[seed][number].append(file_)

        seed_dict = {}
        for seed, models in file_groups.items():
            model_number_file_type_file = {}
            pae_file = None
            if -1 in models:
                for file_ in models[-1]:
                    if file_.pathway.stem.startswith("pae_scores"):
                        pae_file = file_
                        break

            for model_number, files in models.items():
                if model_number == -1:
                    continue
                intermediate_dict = {}
                for file_ in sorted(files, key=lambda x: x.suffix):
                    if file_.pathway.stem.startswith("scores.model"):
                        intermediate_dict["scores"] = file_
                    elif isinstance(file_, CifFile):
                        if file_.pathway.suffix == ".cif":
                            file_.name = f"Chai-1_{seed}_{model_number}"
                            # Chai cif not recognised by pae-viewer, so we load and save
                            file_.to_file(file_.pathway)
                            intermediate_dict["cif"] = file_
                if model_number != -1 and pae_file is not None:
                    new_pae_path = (
                        file_.pathway.parent / f"pae_scores_model_{model_number}.npy"
                    )
                    shutil.copy(pae_file.pathway, new_pae_path)
                    intermediate_dict["pae"] = NpyFile(str(new_pae_path))

                model_number_file_type_file[model_number] = intermediate_dict

            model_number_file_type_file = {
                model_number: model_number_file_type_file[model_number]
                for model_number in sorted(model_number_file_type_file)
            }
            seed_dict[seed] = model_number_file_type_file

        return seed_dict

    def pae_to_af3(self):
        """
        Convert the Chai-1 PAE data to the format expected by AlphaFold3

        Returns:
            None
        """
        new_pae_files: dict[str, list[ConfidenceJsonFile]] = {}
        for seed in self.seeds:
            for i, (pae_file, cif_file) in enumerate(
                zip(self.pae_files[seed], self.cif_files[seed])
            ):
                pae = Af3Pae.from_chai1(
                    pae_file.data[i],
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
                    "scores": self.output[seed][i]["scores"],
                }
                for i, cif_file in enumerate(self.cif_files[seed])
            }
            for seed in self.seeds
        }

    def get_input_fasta(self) -> ChaiFasta:
        """
        Function to get the input fasta file used for the Chai-1 run

        Returns:
            ChaiFasta: ChaiFasta object containing the input fasta file

        """

        ch = ChaiFasta(self.output_dirs[0], self.config, create_files=False)
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
