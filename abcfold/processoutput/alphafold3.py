from pathlib import Path
from typing import Union

from abcfold.processoutput.file_handlers import CifFile, ConfidenceJsonFile


class AlphafoldOutput:
    def __init__(
        self,
        af3_output_dir: Union[str, Path],
        input_params: dict,
        name: str,
    ):
        """
        Object to process the output of an AlphaFold3 run

        Args:
            af3_output_dir (Union[str, Path]): Path to the AlphaFold3 output directory
            input_params (dict): Dictionary containing the input parameters used for the
            AlphaFold3 run
            name (str): Name given to the AlphaFold3 run

        Attributes:
            output_dir (Path): Path to the AlphaFold3 output directory
            input_params (dict): Dictionary containing the input parameters used for the
            AlphaFold3 run
            name (str): Name given to the AlphaFold3 run
            output (dict): Dictionary containing the processed output the contents
            of the AlphaFold3 output directory. The dictionary is structured as follows:

            {
                "seed-1": {
                    1: {
                        "cif": CifFile,
                        "json": ConfidenceJsonFile
                    },
                    2: {
                        "cif": CifFile,
                        "json": ConfidenceJsonFile
                    }
                },
                etc...
            }
            This is different to the boltz and chai equivalent as they do not have seeds
        """
        self.output_dir = Path(af3_output_dir)
        self.input_params = input_params

        if not self.output_dir.name.startswith("alphafold3"):
            self.output_dir = self.output_dir.rename(
                self.output_dir.parent.joinpath(f"alphafold3_{name}")
            )

        self.output = self.process_af3_output()
        self.seeds = list(self.output.keys())
        self.cif_files = {
            seed: [value["cif"] for value in self.output[seed].values()]
            for seed in self.seeds
        }
        self.af3_pae_files = {
            seed: [value["af3_pae"] for value in self.output[seed].values()]
            for seed in self.seeds
        }
        self.scores_files = {
            seed: [value["summary"] for value in self.output[seed].values()]
            for seed in self.seeds
        }
        self.input_json = list(self.output_dir.glob("*_data.json"))[0]

    def process_af3_output(self):
        """
        Process the output of an AlphaFold3 run

        """
        file_groups = {}
        for pathway in self.output_dir.iterdir():
            if pathway.is_dir():
                seed = pathway.name.split("_")[0]
                sample = pathway.name.split("-")[-1]
                if seed not in file_groups:
                    file_groups[seed] = {}
                if not sample.isdigit():
                    continue
                sample = int(sample)
                if sample not in file_groups[seed]:
                    file_groups[seed][sample] = {}
                for file in pathway.iterdir():
                    if file.suffix == ".cif":
                        cif_file = CifFile(str(file), self.input_params)
                        cif_file.name = f"Alphafold3_{seed}_{sample}"
                        cif_file = self.reorder_chains(cif_file)
                        cif_file.to_file(str(file))
                        file_groups[seed][sample]["cif"] = cif_file
                    elif file.suffix == ".json" and "summary" not in file.stem:
                        file_groups[seed][sample]["af3_pae"] = ConfidenceJsonFile(
                            str(file)
                        )
                    elif file.suffix == ".json" and "summary" in file.stem:
                        file_groups[seed][sample]["summary"] = ConfidenceJsonFile(
                            str(file)
                        )
                    else:
                        continue

        for seed in file_groups:
            file_groups[seed] = {
                sample: file_groups[seed][sample]
                for sample in sorted(file_groups[seed])
            }
        return file_groups

    def get_chain_ids(self) -> list:
        ids: list = []
        assert "sequences" in self.input_params
        for sequences in self.input_params["sequences"]:
            for sequence_type, sequence_information in sequences.items():
                if "id" not in sequence_information:
                    continue
                sequence_ids = sequence_information["id"]
                if isinstance(sequence_ids, list):
                    ids.extend(sequence_ids)
                elif isinstance(sequence_ids, str):
                    ids.append(sequence_ids)

        return ids

    def reorder_chains(self, cif_file: CifFile) -> CifFile:
        """
        Function to update the chain order in the CIF file according to the ids in the
        input file

        Args:
            cif_file (CifFile): CifFile object to update the chain labels for

        """

        cif_file.reorder_chains(self.get_chain_ids())
        return cif_file


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
    ao = AlphafoldOutput(
        "/home/etk48667/folding/aaa_bbb/alphafold3_Hello_fold/",
        input_params=input_params,
        name="Hello_fold",
    )

    ao.process_af3_output()
