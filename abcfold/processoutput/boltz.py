import json
import logging
from pathlib import Path
from typing import Union

from abcfold.boltz1.af3_to_boltz1 import BoltzYaml
from abcfold.processoutput.file_handlers import (CifFile, ConfidenceJsonFile,
                                                 FileTypes, ModelCount,
                                                 NpzFile)
from abcfold.processoutput.utils import Af3Pae

logger = logging.getLogger("logger")


class BoltzOutput:
    def __init__(
        self,
        boltz_output_dir: Union[str, Path],
        input_params: dict,
        name: str,
    ):
        """
        Object to process the output of an Boltz-1 run

        Args:
            boltz_output_dir (Union[str, Path]): Path to the Boltz-1 output directory
            input_params (dict): Dictionary containing the input parameters used for the
            Boltz-1 run
            name (str): Name given to the Boltz-1 run

        Attributes:
            output_dir (Path): Path to the Boltz-1 output directory
            input_params (dict): Dictionary containing the input parameters used for the
            Boltz-1 run
            name (str): Name given to the Boltz-1 run
            output (dict): Dictionary containing the processed output the contents
            of the Boltz-1 output directory. The dictionary is structured as follows:

            {
                1: {
                    "pae": NpzFile,
                    "plddt": NpzFile,
                    "pde": NpzFile,
                    "cif": CifFile,
                    "json": ConfidenceJsonFile
                },
                2: {
                    "pae": NpzFile,
                    "plddt": NpzFile,
                    "pde": NpzFile,
                    "cif": CifFile,
                    "json": ConfidenceJsonFile
                },
                ...
            }
            pae_files (list): Ordered list of NpzFile objects containing the PAE data
            plddt_files (list): Ordered list of NpzFile objects containing the PLDDT
            data
            pde_files (list):  Ordered list of NpzFile objects containing the PDE data
            cif_files (list): Ordered list of CifFile objects containing the model data
            scores_files (list): Ordered list of ConfidenceJsonFile objects containing
            the model scores

        """
        self.output_dir = Path(boltz_output_dir)
        self.input_params = input_params
        self.name = name

        if self.output_dir.name.startswith("boltz_results_"):
            self.output_dir = self.output_dir.rename(
                self.output_dir.parent / f"boltz-1_{name}"
            )
        self.yaml_input_obj = self.get_input_yaml()
        self.output = self.process_boltz_output()

        self.pae_files = [value["pae"] for value in self.output.values()]
        self.cif_files = [value["cif"] for value in self.output.values()]
        self.pae_to_af3()
        self.af3_pae_files = [value["af3_pae"] for value in self.output.values()]
        self.plddt_files = [value["plddt"] for value in self.output.values()]
        self.pde_files = [value["pde"] for value in self.output.values()]
        self.scores_files = [value["json"] for value in self.output.values()]

    def process_boltz_output(self):
        """
        Function to process the output of a Boltz-1 run
        """
        file_groups = {}
        for pathway in self.output_dir.rglob("*"):
            number = pathway.stem.split("_model_")[-1]
            if not number.isdigit():
                continue
            number = int(number)

            file_type = pathway.suffix[1:]
            if file_type == FileTypes.NPZ.value:
                file_ = NpzFile(str(pathway))
            elif file_type == FileTypes.CIF.value:
                file_ = CifFile(str(pathway), self.input_params)

            elif file_type == FileTypes.JSON.value:
                file_ = ConfidenceJsonFile(str(pathway))
            else:
                continue
            if number not in file_groups:
                file_groups[number] = [file_]
            else:
                file_groups[number].append(file_)

        model_number_file_type_file = {}
        for model_number, files in file_groups.items():
            intermediate_dict = {}
            for file_ in sorted(files, key=lambda x: x.suffix):
                if file_.pathway.stem.startswith("pae"):
                    intermediate_dict["pae"] = file_
                elif file_.pathway.stem.startswith("plddt"):
                    intermediate_dict["plddt"] = file_
                elif file_.pathway.stem.startswith("pde"):
                    intermediate_dict["pde"] = file_
                elif file_.pathway.suffix == ".cif":
                    file_.name = f"Boltz-1_{model_number}"
                    intermediate_dict["cif"] = file_
                else:
                    intermediate_dict[file_.suffix] = file_

            model_number_file_type_file[model_number] = intermediate_dict

        model_number_file_type_file = {
            key: model_number_file_type_file[key]
            for key in sorted(model_number_file_type_file)
        }
        return model_number_file_type_file

    def add_plddt_to_cif(self):
        """
        Add the PLDDT scores to the B-factors of the CIF files as this is not done
        natively by Boltz-1

        Returns:
            None

        Raises:
            AssertionError: If the length of the PLDDT scores does not match the number
            of residues in the CIF file
        """
        for cif_file, plddt_scores in zip(self.cif_files, self.plddt_files):
            cif_file = self.update_chain_labels(cif_file)
            plddt_score = plddt_scores.data["plddt"]
            if max(plddt_score) <= 1:
                plddt_score = (plddt_score * 100).astype(float)

            chain_lengths = cif_file.chain_lengths(
                mode=ModelCount.RESIDUES, ligand_atoms=True
            )
            print(chain_lengths)

            print(len(plddt_score), sum(chain_lengths.values()))
            assert sum(chain_lengths.values()) == len(plddt_score), "Length mismatch"

            counter = 0
            for chain in cif_file.model[0]:

                # Boltz does ligand plddt per atom so we need to count them separately
                ligand = cif_file.check_ligand(chain)

                for residue in chain:
                    for atom in residue:
                        atom.b_iso = plddt_score[counter]
                        if ligand:
                            counter += 1
                    if not ligand:
                        counter += 1

            assert counter == len(plddt_score), "Length mismatch"
            cif_file.update()

    def pae_to_af3(self):
        """
        Convert the PAE data from Boltz-1 to the format used by Alphafold3

        Returns:
            None
        """
        for i, (pae_file, cif_file) in enumerate(zip(self.pae_files, self.cif_files)):
            pae = Af3Pae.from_boltz1(
                pae_file.data,
                cif_file,
            )

            out_name = cif_file.pathway.parent.joinpath(
                cif_file.pathway.stem + "_af3_pae.json"
            )

            with open(out_name, "w") as f:
                json.dump(pae.scores, f)

            self.output[i]["af3_pae"] = ConfidenceJsonFile(out_name)

    def update_chain_labels(self, cif_file) -> CifFile:
        """
        Function to update the chain labels in the CIF file

        Args:
            cif_file (CifFile): CifFile object to update the chain labels for

        """
        cif_file.relabel_chains(
            self.yaml_input_obj.chain_ids, self.yaml_input_obj.id_links
        )
        return cif_file

    def get_input_yaml(self) -> BoltzYaml:
        """
        Get the input yaml file used for the Boltz-1 run

        Returns:
            BoltzYaml: Object containing the input yaml file
        """

        by = BoltzYaml(self.output_dir, create_files=False)
        by.json_to_yaml(self.input_params)

        return by


if __name__ == "__main__":
    input_params = {
        "name": "Hello_fold",
        "modelSeeds": [42],
        "sequences": [
            {
                "protein": {
                    "id": "A",
                    "sequence": "PVLSCGEWQL",
                    "modifications": [
                        {"ptmType": "HY3", "ptmPosition": 1},
                        {"ptmType": "P1L", "ptmPosition": 5},
                    ],
                }
            },
            {"protein": {"id": "B", "sequence": "QIQLVQSGPELKKPGET"}},
            {"protein": {"id": "C", "sequence": "DVLMIQTPLSLPVS"}},
            {"ligand": {"id": ["F"], "ccdCodes": ["ATP"]}},
            {"ligand": {"id": "I", "ccdCodes": ["NAG", "FUC", "FUC"]}},
            {"dna": {"id": ["D", "K"], "sequence": "AGCT"}},
            {"rna": {"id": "L", "sequence": "AGCU"}},
            {"ligand": {"id": "Z", "smiles": "CC(=O)OC1C[NH+]2CCC1CC2"}},
        ],
        "bondedAtomPairs": [
            [["A", 1, "CA"], ["B", 1, "CA"]],
            [["C", 7, "CA"], ["A", 10, "CA"]],
            [["I", 1, "O3"], ["I", 2, "C1"]],
            [["I", 2, "C1"], ["I", 3, "C1"]],
        ],
        "dialect": "alphafold3",
        "version": 2,
    }
    bo = BoltzOutput(
        "/home/etk48667/folding/aaa_bbb_ccc/boltz-1_Hello_fold",
        input_params,
        "Hello_fold",
    )
    bo.add_plddt_to_cif()
