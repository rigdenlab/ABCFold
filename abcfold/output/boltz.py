import logging
from pathlib import Path
from typing import Union

from abcfold.boltz.af3_to_boltz import BoltzYaml
from abcfold.output.file_handlers import (CifFile, ConfidenceJsonFile,
                                          FileTypes, ModelCount, NpzFile)
from abcfold.output.utils import Af3Pae

logger = logging.getLogger("logger")


class BoltzOutput:
    def __init__(
        self,
        boltz_output_dirs: list[Union[str, Path]],
        input_params: dict,
        name: str,
        save_input: bool = False,
    ):
        """
        Object to process the output of an Boltz run

        Args:
            boltz_output_dirs (list[Union[str, Path]]): Path to the Boltz
            output directory
            input_params (dict): Dictionary containing the input parameters used for the
            Boltz run
            name (str): Name given to the Boltz run
            save_input (bool): If True, Boltz was run with the save_input flag

        Attributes:
            output_dirs (list): List of paths to the Boltz output directory(s)
            input_params (dict): Dictionary containing the input parameters used for the
            Boltz run
            name (str): Name given to the Boltz run
            output (dict): Dictionary containing the processed output the contents
            of the Boltz output directory(s). The dictionary is structured as follows:

            {
                "seed-1": {
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
                },
                etc...
            }
            pae_files (list): Ordered list of NpzFile objects containing the PAE data
            plddt_files (list): Ordered list of NpzFile objects containing the PLDDT
            data
            pde_files (list):  Ordered list of NpzFile objects containing the PDE data
            cif_files (list): Ordered list of CifFile objects containing the model data
            scores_files (list): Ordered list of ConfidenceJsonFile objects containing
            the model scores
        """
        self.output_dirs = [Path(x) for x in boltz_output_dirs]
        self.input_params = input_params
        self.name = name
        self.save_input = save_input

        parent_dir = self.output_dirs[0].parent
        new_parent = parent_dir / f"boltz_{self.name}"
        new_parent.mkdir(parents=True, exist_ok=True)

        if self.save_input:
            boltz_yaml = list(parent_dir.glob("*.yaml"))[0]
            if boltz_yaml.exists():
                boltz_yaml.rename(new_parent / "boltz_input.yaml")

            boltz_msas = list(parent_dir.glob("*.a3m"))
            if boltz_msas:
                for boltz_msa in boltz_msas:
                    if boltz_msa.exists():
                        boltz_msa.rename(new_parent / boltz_msa.name)

        new_output_dirs = []
        for output_dir in self.output_dirs:
            if output_dir.name.startswith("boltz_results_"):
                new_path = new_parent / output_dir.name
                output_dir.rename(new_path)
                new_output_dirs.append(new_path)
            else:
                new_output_dirs.append(output_dir)
        self.output_dirs = new_output_dirs

        self.yaml_input_obj = self.get_input_yaml()

        self.output = self.process_boltz_output()
        self.seeds = list(self.output.keys())
        self.pae_files = {
            seed: [value["pae"] for value in self.output[seed].values()]
            for seed in self.seeds
        }
        self.cif_files = {
            seed: [value["cif"] for value in self.output[seed].values()]
            for seed in self.seeds
        }
        self.plddt_files = {
            seed: [value["plddt"] for value in self.output[seed].values()]
            for seed in self.seeds
        }
        self.pde_files = {
            seed: [value["pde"] for value in self.output[seed].values()]
            for seed in self.seeds
        }
        self.scores_files = {
            seed: [value["json"] for value in self.output[seed].values()]
            for seed in self.seeds
        }
        self.pae_to_af3()
        self.af3_pae_files = {
            seed: [value["af3_pae"] for value in self.output[seed].values()]
            for seed in self.seeds
        }

    def process_boltz_output(self):
        """
        Function to process the output of a Boltz run
        """
        file_groups: dict[str, dict[int, list]] = {}
        for pathway in self.output_dirs:
            seed = pathway.name.split("_")[-1]
            if seed not in file_groups:
                file_groups[seed] = {}

            for output in pathway.rglob("*"):
                number = None
                number_str = output.stem.split("_model_")[-1]
                if not number_str.isdigit():
                    continue
                number = int(number_str)

                if number is None:
                    continue

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
                if number not in file_groups[seed]:
                    file_groups[seed][number] = [file_]
                else:
                    file_groups[seed][number].append(file_)

        seed_dict = {}
        for seed, models in file_groups.items():
            model_number_file_type_file = {}
            for model_number, files in models.items():
                intermediate_dict = {}
                for file_ in sorted(files, key=lambda x: x.suffix):
                    if file_.pathway.stem.startswith("pae"):
                        intermediate_dict["pae"] = file_
                    elif file_.pathway.stem.startswith("plddt"):
                        intermediate_dict["plddt"] = file_
                    elif file_.pathway.stem.startswith("pde"):
                        intermediate_dict["pde"] = file_
                    elif isinstance(file_, CifFile):
                        if file_.pathway.suffix == ".cif":
                            file_.name = f"Boltz_{seed}_{model_number}"
                            file_ = self.update_chain_labels(file_)
                            intermediate_dict["cif"] = file_
                    else:
                        intermediate_dict[file_.suffix] = file_

                model_number_file_type_file[model_number] = intermediate_dict

            model_number_file_type_file = {
                key: model_number_file_type_file[key]
                for key in sorted(model_number_file_type_file)
            }
            seed_dict[seed] = model_number_file_type_file

        return seed_dict

    def add_plddt_to_cif(self):
        """
        Add the PLDDT scores to the B-factors of the CIF files as this is not done
        natively by Boltz

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
                mode=ModelCount.RESIDUES, ligand_atoms=True, ptm_atoms=True
            )

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
        Convert the PAE data from Boltz to the format used by Alphafold3

        Returns:
            None
        """
        new_pae_files: dict[str, list[ConfidenceJsonFile]] = {}
        for seed in self.seeds:
            for (pae_file, cif_file) in zip(self.pae_files[seed], self.cif_files[seed]):
                pae = Af3Pae.from_boltz(
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
                    "json": self.output[seed][i]["json"],
                }
                for i, cif_file in enumerate(self.cif_files[seed])
            }
            for seed in self.seeds
        }

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
        Get the input yaml file used for the Boltz run

        Returns:
            BoltzYaml: Object containing the input yaml file
        """

        by = BoltzYaml(self.output_dirs[0], create_files=False)
        by.json_to_yaml(self.input_params)

        return by
