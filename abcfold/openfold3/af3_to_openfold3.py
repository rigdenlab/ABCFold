import json
import logging
import uuid
from pathlib import Path
from typing import Any, Dict, Union

logger = logging.getLogger("logger")


class OpenfoldJson:
    """
    Object to convert an AlphaFold3 json file to a OpenFold 3 JSON file.
    """

    def __init__(self, working_dir: Union[str, Path],
                 create_files: bool = True):
        self.working_dir = working_dir
        self.seeds: list = [42]
        self.__ids: Dict = {}
        self.__create_files = create_files
        self.name = ""
        self.openfold_dict: Dict = {"queries": {}}
        self._use_templates = False

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

    def _write_template_cifs(self, templates) -> list:
        """Write inline mmCIF templates to .cif files for CIF-direct mode.

        Args:
            templates (list): AF3-style template entries, each carrying an
                inline ``mmcif`` string.

        Returns:
            list: Paths to the written .cif files, one per template with an
            mmCIF payload (skips any malformed/empty entries).
        """
        cif_paths: list = []
        tmpl_dir = Path(self.working_dir) / f"templates_{uuid.uuid4().hex}"
        tmpl_dir.mkdir(parents=True, exist_ok=True)
        for idx, template in enumerate(templates):
            mmcif = template.get("mmcif") if isinstance(template, dict) else None
            if not mmcif:
                continue
            cif_file = tmpl_dir / f"template_{idx}.cif"
            cif_file.write_text(mmcif)
            cif_paths.append(cif_file.resolve().as_posix())
        return cif_paths

    def json_to_json(
        self,
        json_file_or_dict: Union[dict, str, Path],
    ):
        """
        Main function to convert an AF3 json file or dict to a OpenFold 3 json string

        Args:
            json_file_or_dict (Union[dict, str, Path]): json file or dict

        Returns:
            Dict: OpenFold 3 dictionary
        """
        logger.info("Converting input json to a OpenFold 3 compatible json file")
        if isinstance(json_file_or_dict, str) or isinstance(json_file_or_dict, Path):
            with open(json_file_or_dict, "r") as f:
                json_dict = json.load(f)
        else:
            json_dict = json_file_or_dict

        query_flags_set = False
        openfold_sequences = []
        for key, value in json_dict.items():
            if key == "name":
                self.name = value
                self.openfold_dict["queries"][self.name] = {}
            if key == "modelSeeds":
                if isinstance(value, list):
                    self.seeds = value
                elif isinstance(value, int):
                    self.seeds = [value]

            if key == "sequences":
                for entry in value:
                    if "protein" in entry:
                        chain_entry = self.convert_protein(entry["protein"])
                    elif "rna" in entry:
                        chain_entry = self.convert_rna(entry["rna"])
                    elif "dna" in entry:
                        chain_entry = self.convert_dna(entry["dna"])
                    elif "ligand" in entry:
                        chain_entry = self.convert_ligand(entry["ligand"])
                    if chain_entry:
                        openfold_sequences.append(chain_entry)

                        if (
                            "main_msa_file_paths" in chain_entry
                            and chain_entry["main_msa_file_paths"]
                        ):
                            if not query_flags_set:
                                q = self.openfold_dict["queries"][self.name]
                                q["use_msas"] = True
                                q["use_main_msas"] = True
                                q["use_paired_msas"] = False
                                query_flags_set = True

        self.openfold_dict["queries"][self.name]["chains"] = openfold_sequences

        return self.openfold_dict

    def convert_protein(self, seq_dict) -> Dict[str, Any]:
        sequence = seq_dict["sequence"]
        chain_ids = seq_dict.get("id", [])

        protein_chain = {
            "molecule_type": "protein",
            "chain_ids": chain_ids,
            "sequence": sequence,
        }

        modifications = seq_dict.get("modifications")
        if modifications:
            protein_chain["non_canonical_residues"] = {}
            for mod in modifications:
                loc = str(mod['ptmPosition'])
                ptm_type = mod['ptmType']
                protein_chain["non_canonical_residues"][loc] = ptm_type

        templates = seq_dict.get("templates")
        if templates and self.__create_files:
            cif_paths = self._write_template_cifs(templates)
            if cif_paths:
                protein_chain["template_cif_paths"] = cif_paths
                self._use_templates = True

        unpaired_msa = seq_dict.get("unpairedMsa")
        msa_dir = Path(self.working_dir) / uuid.uuid4().hex
        if unpaired_msa and self.__create_files:
            msa_out = msa_dir / "colabfold_main.a3m"
            if not msa_dir.exists():
                msa_dir.mkdir(parents=True, exist_ok=True)
            self.msa_to_file(
                unpaired_msa,
                msa_out
            )
            protein_chain["main_msa_file_paths"] = [
                msa_dir.resolve().as_posix()
            ]

        return protein_chain

    def convert_rna(self, seq_dict) -> Dict[str, Any]:
        sequence = seq_dict["sequence"]
        chain_ids = seq_dict.get("id", [])

        rna_chain = {
            "molecule_type": "rna",
            "chain_ids": chain_ids,
            "sequence": sequence,
        }

        modifications = seq_dict.get("modifications")
        if modifications:
            rna_chain["non_canonical_residues"] = {}
            for mod in modifications:
                loc = str(mod['basePosition'])
                ptm_type = mod['modificationType']
                rna_chain["non_canonical_residues"][loc] = ptm_type
        modifications = seq_dict.get("modifications")

        return rna_chain

    def convert_dna(self, seq_dict) -> Dict[str, Any]:
        sequence = seq_dict["sequence"]
        chain_ids = seq_dict.get("id", [])

        dna_chain = {
            "molecule_type": "dna",
            "chain_ids": chain_ids,
            "sequence": sequence,
        }

        modifications = seq_dict.get("modifications")
        if modifications:
            dna_chain["non_canonical_residues"] = {}
            for mod in modifications:
                loc = str(mod['basePosition'])
                ptm_type = mod['modificationType']
                dna_chain["non_canonical_residues"][loc] = ptm_type
        modifications = seq_dict.get("modifications")

        return dna_chain

    def convert_ligand(self, seq_dict) -> Dict[str, Any]:
        chain_ids = seq_dict.get("id", [])

        ligand_chain = {
            "molecule_type": "ligand",
            "chain_ids": chain_ids,
        }

        if "ccdCodes" in seq_dict:
            ligand_id = seq_dict["ccdCodes"][0]
            ligand_chain["ccd_codes"] = ligand_id
        else:
            ligand_id = seq_dict["smiles"]
            ligand_chain["smiles"] = ligand_id

        return ligand_chain

    def write_json(self, out_file: Union[str, Path]):
        """
        Write the OpenFold 3 json to a file

        Args:
            out_file (Union[str, Path]): output file path

        Returns:
            None
        """

        with open(out_file, "w") as f:
            json.dump(self.openfold_dict, f, indent=4)

    def write_yaml(self, out_file: Union[str, Path]):
        """
        Write the OpenFold 3 runner yaml file

        Args:
            out_file (Union[str, Path]): output file path

        Returns:
            None
        """
        seeds_str = f"[{', '.join(str(s) for s in self.seeds)}]"
        use_templates = self._use_templates

        lines = [
            "model_update:",
            "  presets:",
            "    - predict",
            "    - low_mem",
            "  custom:",
            "    settings:",
            "      memory:",
            "        eval:",
            "          use_cueq_triangle_kernels: true",
            "          use_deepspeed_evo_attention: false",
            "experiment_settings:",
            "  mode: predict",
            f"  seeds: {seeds_str}",
            f"  use_templates: {str(use_templates).lower()}",
        ]

        yaml_string = "\n".join(lines)

        try:
            out_path = Path(out_file)
            if self.__create_files:
                out_path.write_text(yaml_string)
        except Exception:
            logger.debug("Could not write yaml to %s", out_file)

        return yaml_string
