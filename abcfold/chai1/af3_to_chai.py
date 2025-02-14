import hashlib
import json
import logging
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as _  # noqa F401
import pandas as pd
import requests

from abcfold.processoutput.atoms import VANDERWALLS

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


ATOMS_NAMES = sorted(list(VANDERWALLS.keys()), key=len, reverse=True)


class ChaiFasta:

    def __init__(self, working_dir: Union[str, Path], create_files: bool = True):
        """
        Object to convert an AlphaFold3 json to a fasta file compatible with CHAI-1

        Args:
            working_dir (Union[str, Path]): working directory to store the fasta file

        Attributes:
            working_dir (Path): working directory to store the fasta file
            fasta (Path): path to the fasta file
            constraints (Path): path to the constraints file
            msa_file (Optional[Union[str, Path]]): path to the msa file


        """
        self.working_dir = Path(working_dir)
        self.fasta = Path(working_dir) / "chai1.fasta"
        self.constraints = Path(working_dir) / "chai1_constraints.csv"
        self.msa_file: Optional[Union[str, Path]] = None
        self.__ids: List[Union[str, int]] = []
        self.__create_files = create_files

    @property
    def chain_ids(self):
        return self.__ids

    def bonded_pairs_to_file(self, bonded_pairs, fasta_data: dict):
        """
        Converts bonded pairs to a csv file compatible with CHAI-1

        Args:
            bonded_pairs (list): list of bonded pairs in AF3 format
            fasta_data (dict): dictionary with sequence data relating to IDs

        Returns:
            None
        """

        constraints_headers = [
            "chainA",
            "res_idxA",
            "chainB",
            "res_idxB",
            "connection_type",
            "confidence",
            "min_distance_angstrom",
            "max_distance_angstrom",
            "comment",
            "restraint_id",
        ]
        df = pd.DataFrame(columns=constraints_headers)

        for i, bonded_pair in enumerate(bonded_pairs):
            pair_data = []
            if (
                fasta_data[bonded_pair[0][0]] == "CCD-CODE_PLACEHOLDER"
                or fasta_data[bonded_pair[1][0]] == "CCD-CODE_PLACEHOLDER"
            ):
                logger.warning(
                    "Currently Chai-1 does not support most ccdcode ligands\
 in bonded pairs. This is something that we will be keeping in mind for future\
 updates"
                )
                continue
            if (
                fasta_data[bonded_pair[0][0]] == "LIGAND_PLACEHOLDER"
                or fasta_data[bonded_pair[1][0]] == "LIAGND_PLACEHOLDER"
            ):
                logger.warning(
                    "SMILES Ligand bonded paris are not implemented yet, please \
check back for updates"
                )
                continue

            for pair in bonded_pair:

                chain_id = pair[0]
                seq_idx = pair[1]
                pair_data.append(chain_id)
                res = fasta_data[chain_id][seq_idx - 1]

                pair_data.append(f"{res}{seq_idx}@{pair[2]}")

            row_data = [
                pair_data[0],
                pair_data[1],
                pair_data[2],
                pair_data[3],
                "contact",
                1.0,
                0.0,
                5.5,
                "Covalent Bond",
                f"restraint_{i}",
            ]
            df.loc[i] = row_data

        if not df.empty and self.__create_files:
            df.to_csv(self.constraints, sep=",", index=False)

    def msa_to_file(self, msa: str, file_path: Union[str, Path]):
        """
        Takes an msa string, converts it to pqt format and writes it to a file

        Args:
            msa (str): msa string (a3m format)
            file_path (Union[str, Path]): file path to write the msa to

        Returns:
            None
        """
        try:
            from chai_lab.data.parsing.msas.aligned_pqt import \
                merge_multi_a3m_to_aligned_dataframe
            from chai_lab.data.parsing.msas.data_source import MSADataSource
        except ImportError:

            logger.error(
                "Chai_lab didn't install correctly and the module is not available, \
please try running the job again or install chai_lab directly using 'pip \
install chai_lab'"
            )
            raise ImportError()

        # Convert msa to CHAI-1 format with additional MSA source information
        with tempfile.NamedTemporaryFile(suffix=".a3m", mode="w") as f:
            f.write(msa)
            f.flush()
            df = merge_multi_a3m_to_aligned_dataframe(
                msa_a3m_files={Path(f.name): MSADataSource.UNIPROT},
                insert_keys_for_sources="uniprot",
            )

        # Write the dataframe to a parquet file
        if not df.empty and self.__create_files:
            df.to_parquet(file_path)

    def json_to_fasta(self, json_file_or_dict: Union[dict, str, Path]):
        """
        Main function to convert an AlphaFold3 json to a fasta file
        compatible with CHAI-1

        Args:
            json_file_or_dict (Union[dict, str, Path]):
            json file or dictionary

        Returns:
            None
        """
        logger.info("Converting input json to a Chai-1 compatible fasta file")
        if isinstance(json_file_or_dict, str) or isinstance(json_file_or_dict, Path):
            with open(json_file_or_dict, "r") as f:
                json_dict = json.load(f)
        else:
            json_dict = json_file_or_dict

        fasta_data: Dict[str, str] = {}

        with open(self.fasta, "w") as f:
            for seq in json_dict["sequences"]:
                if "protein" in seq:
                    protein_str, fasta_data = self.add_protein(seq, fasta_data)
                    f.write(protein_str)
                if "rna" in seq:
                    nucleotide_str, fasta_data = self.add_nucleotide(
                        seq, "rna", fasta_data
                    )
                    f.write(nucleotide_str)
                if "dna" in seq:
                    nucleotide_str, fasta_data = self.add_nucleotide(
                        seq, "dna", fasta_data
                    )
                    f.write(nucleotide_str)
                if "ligand" in seq:
                    ligand_str = self.add_ligand(seq, fasta_data)
                    f.write(ligand_str)

        if "bondedAtomPairs" in json_dict.keys() and self.__create_files:
            if isinstance(json_dict["bondedAtomPairs"], list):
                bonded_pairs = json_dict["bondedAtomPairs"]
                self.bonded_pairs_to_file(bonded_pairs, fasta_data)

        if not self.__create_files:
            self.fasta.unlink()

        self.__ids = list(fasta_data.keys())

    def add_protein(self, seq: dict, fasta_data: dict):
        prot_id = seq["protein"]["id"]
        if isinstance(prot_id, list):
            protein_str = ""
            for i in seq["protein"]["id"]:
                temp_prot_str, fasta_data = self._add_protein(seq, i, fasta_data)
                protein_str += temp_prot_str
        else:
            protein_str, fasta_data = self._add_protein(seq, prot_id, fasta_data)
        return protein_str, fasta_data

    def _add_protein(self, seq: dict, prot_id: str, fasta_data: dict):
        protein_str = f">protein|{prot_id}\n{seq['protein']['sequence']}\n"
        sequence = seq["protein"]["sequence"]
        if "unpairedMsa" in seq["protein"].keys():
            seq_hash = hashlib.sha256(sequence.upper().encode()).hexdigest()
            pqt_path = Path(self.working_dir) / f"{seq_hash}.aligned.pqt"
            msa = seq["protein"]["unpairedMsa"]

            (
                self.msa_to_file(msa=msa, file_path=pqt_path)
                if self.__create_files
                else None
            )
        fasta_data[prot_id] = sequence

        return protein_str, fasta_data

    def add_nucleotide(self, seq: dict, seq_type: str, fasta_data: dict):
        nucl_id = seq[seq_type]["id"]

        if isinstance(nucl_id, list):
            nucleotide_str = ""
            for i in seq[seq_type]["id"]:
                temp_nucl_str, fasta_data = self._add_nucleotide(
                    seq, seq_type, i, fasta_data
                )
                nucleotide_str += temp_nucl_str
        else:
            nucleotide_str, fasta_data = self._add_nucleotide(
                seq, seq_type, nucl_id, fasta_data
            )

        return nucleotide_str, fasta_data

    def _add_nucleotide(self, seq: dict, seq_type: str, nucl_id: str, fasta_data: dict):
        nucleotide_str = f">{seq_type}|{nucl_id}\n{seq[seq_type]['sequence']}\n"
        fasta_data[nucl_id] = seq[seq_type]["sequence"]
        return nucleotide_str, fasta_data

    def ccd_to_smiles(self, ccd_id: str):

        assert isinstance(ccd_id, str), "CCD ID must be a string"
        logger.info(f"CCD code found in input: {ccd_id}")
        logger.info("Chai-1 currently only supports SMILES strings for ligands")
        logger.info("Attempting to retrieve SMILES from CCD")
        url = f"http://cactus.nci.nih.gov/chemical/structure/{ccd_id}/smiles"
        response = requests.get(url)
        if response.status_code == 200:
            ccd_data = response.text
            logger.info(f"SMILES retrieved: {ccd_data}")
            return ccd_data
        else:
            logger.warning(f"Could not retrieve SMILES for {ccd_id}")
            return None

    def add_ligand(self, seq: dict, fasta_data: dict):
        lig_id = seq["ligand"]["id"]
        ligand_str = ""
        if "ccdCodes" in seq["ligand"]:
            if isinstance(lig_id, str):
                lig_id = [lig_id]

            for lig in lig_id:
                if isinstance(seq["ligand"]["ccdCodes"], str):
                    ccd_codes = [seq["ligand"]["ccdCodes"]]
                else:
                    ccd_codes = seq["ligand"]["ccdCodes"]
                ligand_str += (
                    f">protein|{lig}\n{''.join([f'({ccd})' for ccd in ccd_codes])}\n"
                )
                fasta_data[lig] = "CCD-CODE_PLACEHOLDER"

            # if isinstance(lig_id, list):
            #     for i in lig_id:
            #         ccd_code = (
            #             seq["ligand"]["ccdCodes"][0]
            #             if isinstance(seq["ligand"]["ccdCodes"], list)
            #             else seq["ligand"]["ccdCodes"]
            #         )

            #         smile = self.ccd_to_smiles(ccd_code)
            #         if smile:
            #             ligand_str += f">ligand|{i}\n{smile}\n"
            # else:
            #     ccd_code = (
            #         seq["ligand"]["ccdCodes"][0]
            #         if isinstance(seq["ligand"]["ccdCodes"], list)
            #         else seq["ligand"]["ccdCodes"]
            #     )
            #     smile = self.ccd_to_smiles(ccd_code)
            #     if smile:
            #         ligand_str = f">ligand|{lig_id}\n{smile}\n"
        if "smiles" in seq["ligand"]:
            if isinstance(lig_id, list):
                for i in seq["ligand"]["id"]:
                    ligand_str += f">ligand|{i}\n{seq['ligand']['smiles']}\n"
                    fasta_data[i] = "SMILES_PLACEHOLDER"
            else:
                ligand_str = f">ligand|{lig_id}\n{seq['ligand']['smiles']}\n"
                fasta_data[lig_id] = "SMILES_PLACEHOLDER"

        return ligand_str

    def get_atom_name(self, atom: str) -> str:
        for name in ATOMS_NAMES:
            if atom.startswith(name):
                return name
        return ""
