import logging
from pathlib import Path
from typing import Union

from abcfold.output.file_handlers import CifFile, ConfidenceJsonFile
from abcfold.output.utils import Af3Pae

logger = logging.getLogger("logger")


class SeedSampleOutput:
    """
    Shared output handling for backends that write one directory per seed and
    name their files ``<stem>_sample-<n>``.

    Subclasses set ``results_prefix`` to the directory prefix the runner wrote,
    ``label`` to the name used for the collected models, and may override
    ``read_cif`` when a backend stores pLDDT on a different scale.
    """

    results_prefix: str = ""
    label: str = ""
    sample_token: str = "_sample-"
    plddt_is_fraction: bool = False

    def __init__(
        self,
        output_dirs: list[Union[str, Path]],
        input_params: dict,
        name: str,
        save_input: bool = False,
    ):
        """
        Args:
            output_dirs (list[Union[str, Path]]): Paths to the run's output
            directories, one per seed
            input_params (dict): Dictionary containing the input parameters used
            for the run
            name (str): Name given to the run
            save_input (bool): If True, the backend was run with the save_input
            flag

        Attributes:
            output (dict): seed -> model number -> {"cif", "af3_pae", "scores"}
            cif_files (dict): seed -> ordered list of CifFile objects
            scores_files (dict): seed -> ordered list of ConfidenceJsonFile
            pae_files (dict): seed -> ordered list of ConfidenceJsonFile
        """
        self.output_dirs = [Path(x) for x in output_dirs]
        self.input_params = input_params
        self.name = name
        self.save_input = save_input

        parent_dir = self.output_dirs[0].parent
        new_parent = parent_dir / f"{self.label}_{self.name}"
        new_parent.mkdir(parents=True, exist_ok=True)

        if self.save_input:
            for json_file in parent_dir.glob("*.json"):
                json_file.rename(new_parent / json_file.name)
            for msa_file in parent_dir.glob("*/*.a3m"):
                msa_file.rename(new_parent / msa_file.name)

        new_output_dirs = []
        for output_dir in self.output_dirs:
            if output_dir.name.startswith(self.results_prefix):
                new_path = new_parent / output_dir.name
                output_dir.rename(new_path)
                new_output_dirs.append(new_path)
            else:
                new_output_dirs.append(output_dir)
        self.output_dirs = new_output_dirs

        self.output = self.process_output()
        self.seeds = list(self.output.keys())

        self.cif_files = {
            seed: [value["cif"] for value in self.output[seed].values()]
            for seed in self.seeds
        }
        self.scores_files = {
            seed: [value["score"] for value in self.output[seed].values()]
            for seed in self.seeds
        }
        self.pae_files = {
            seed: [value["pae"] for value in self.output[seed].values()]
            for seed in self.seeds
        }
        self.pae_to_af3()
        self.af3_pae_files = {
            seed: [value["af3_pae"] for value in self.output[seed].values()]
            for seed in self.seeds
        }

    def read_cif(self, pathway: Path) -> CifFile:
        if self.plddt_is_fraction:
            return CifFile.from_rosettafold(pathway, self.input_params)
        return CifFile(str(pathway), self.input_params)

    def seed_for(self, output_dir: Path, pathway: Path) -> str:
        """
        Return the seed a produced file belongs to.

        The default reads it from the run directory name; backends that write a
        single directory covering several seeds override this.
        """
        return output_dir.name.split("-")[-1].split("_")[-1]

    def sample_number(self, pathway: Path):
        number_str = pathway.stem.split(self.sample_token)[-1].split("_")[0]
        return int(number_str) if number_str.isdigit() else None

    def pae_data(self, scores: dict):
        """Return the PAE matrix from a backend's confidence file."""
        return scores.get("pae")

    def process_output(self):
        file_groups: dict[str, dict[int, list]] = {}
        for pathway in self.output_dirs:
            for output in pathway.rglob("*"):
                if output.is_dir():
                    continue
                number = self.sample_number(output)
                if number is None:
                    continue
                seed = self.seed_for(pathway, output)
                file_groups.setdefault(seed, {})

                file_: Union[CifFile, ConfidenceJsonFile]
                if output.suffix == ".cif":
                    if output.stem.endswith("_fixed"):
                        continue
                    file_ = self.read_cif(output)
                elif output.suffix == ".json":
                    file_ = ConfidenceJsonFile(str(output))
                else:
                    continue

                file_groups[seed].setdefault(number, []).append(file_)

        seed_dict = {}
        for seed, models in file_groups.items():
            collected = {}
            for model_number, files in sorted(models.items()):
                intermediate: dict[str, Union[CifFile, ConfidenceJsonFile]] = {}
                for file_ in files:
                    stem = file_.pathway.stem
                    if isinstance(file_, CifFile):
                        file_.name = f"{self.label}_{seed}_{model_number}"
                        intermediate["cif"] = file_
                    elif "summary" in stem:
                        intermediate["score"] = file_
                    else:
                        intermediate["pae"] = file_
                if "cif" in intermediate:
                    collected[model_number] = intermediate
            seed_dict[seed] = collected

        return seed_dict

    def pae_to_af3(self):
        """
        Convert each backend's PAE data to the format used by AlphaFold3

        Returns:
            None
        """
        new_pae_files: dict[str, list[ConfidenceJsonFile]] = {}
        for seed in self.seeds:
            for pae_file, cif_file in zip(self.pae_files[seed], self.cif_files[seed]):
                matrix = self.pae_data(pae_file.data)
                if matrix is None:
                    logger.warning(
                        "No PAE data found for %s seed %s", self.label, seed
                    )
                    continue
                pae = Af3Pae.from_rosettafold3({"pae": matrix}, cif_file)
                out_name = pae_file.pathway
                pae.to_file(out_name)
                new_pae_files.setdefault(seed, []).append(ConfidenceJsonFile(out_name))

        self.output = {
            seed: {
                i: {
                    "cif": cif_file,
                    "af3_pae": new_pae_files[seed][i],
                    "scores": self.output[seed][i]["score"],
                }
                for i, cif_file in enumerate(self.cif_files[seed])
                if seed in new_pae_files and i < len(new_pae_files[seed])
            }
            for seed in self.seeds
        }
