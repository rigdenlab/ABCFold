import json
import logging
from abc import ABC
from enum import Enum
from itertools import groupby
from operator import itemgetter
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from Bio.PDB import MMCIFIO, Chain, MMCIFParser
from Bio.PDB.Atom import Atom
from Bio.PDB.kdtrees import KDTree
from Bio.SeqUtils import seq1

logger = logging.getLogger("logger")


class FileTypes(Enum):
    """
    Enum class for the different file types
    """

    NPZ = "npz"
    NPY = "npy"
    CIF = "cif"
    JSON = "json"

    @classmethod
    def values(cls):
        return [value.value for value in cls.__members__.values()]


class ModelCount(Enum):
    """
    Enum class for the different model count types
    """

    ALL = "all"
    RESIDUES = "residues"

    @classmethod
    def values(cls):
        return [value.value for value in cls.__members__.values()]


class ResidueCountType(Enum):
    """
    Enum class for the different residue count types
    """

    AVERAGE = "average"
    CARBONALPHA = "carbonalpha"

    @classmethod
    def values(cls):
        return [value.value for value in cls.__members__.values()]


class FileBase(ABC):
    """
    Abstract base class for the different file types
    """

    def __init__(self, pathway: Union[str, Path]):
        self.pathway = Path(pathway)
        self.suffix = self.pathway.suffix[1:]

    def __str__(self):
        return str(self.pathway)

    def __repr__(self):
        return f"{self.__class__.__name__}({self.pathway})"


class NpzFile(FileBase):

    def __init__(self, npz_file: Union[str, Path]):
        """
        Object to handle npz files

        Args:
            npz_file (Union[str, Path]): Path to the npz file

        Attributes:
            npz_file (Path): Path to the npz file
            data (dict): Dictionary containing the data from the npz file

        """
        super().__init__(npz_file)
        self.npz_file = Path(npz_file)
        self.data = self.load_npz_file()

    def load_npz_file(self) -> dict:
        return dict(np.load(self.npz_file))


class NpyFile(FileBase):
    def __init__(self, npy_file: Union[str, Path]):
        """
        Object to handle npy files

        Args:
            npy_file (Union[str, Path]): Path to the npy file

        Attributes:
            npy_file (Path): Path to the npy file
            data (np.ndarray): Numpy array containing the data from the np
        """

        super().__init__(npy_file)
        self.npy_file = Path(npy_file)
        self.data = self.load_npy_file()

    def load_npy_file(self) -> np.ndarray:
        return np.load(self.npy_file)


class CifFile(FileBase):

    def __init__(self, cif_file: Union[str, Path], input_params: Optional[dict] = None):
        """
        Object to handle cif files

        Args:
            cif_file (Union[str, Path]): Path to the cif file
            input_params (Optional[dict]): Dictionary containing the input parameters
            used for the model. This is used to distinguish between ligands and
            sequences

        Attributes:
            cif_file (Path): Path to the cif file
            model (Structure): BioPython structure object containing the model
            atom_plddt_per_chain (dict): Dictionary containing the pLDDT scores for
            each atom
            residue_plddt_per_chain (dict): Dictionary containing the pLDDT scores for
            each residue
            plddts (list): List containing the pLDDT scores for each atom
            residue_plddts (list): List containing the pLDDT scores for each residue
            name (str): Name given to the model
        """
        if input_params is None:
            self.input_params = {}
        else:
            self.input_params = input_params

        super().__init__(cif_file)
        self.cif_file = Path(cif_file)
        self.clashes = 0
        self.model = self.load_cif_file()
        self.atom_plddt_per_chain = self.get_plddt_per_atom()
        self.residue_plddt_per_chain = self.get_plddt_per_residue()
        self.ligand_plddt = self.get_plddt_per_ligand()
        self.__plddts = [
            plddts for plddts in self.atom_plddt_per_chain.values() for plddts in plddts
        ]
        self.__residue_plddts = [
            plddts
            for plddts in self.residue_plddt_per_chain.values()
            for plddts in plddts
        ]
        self.__ligand_plddts = [
            plddts for plddts in self.ligand_plddt.values() for plddts in plddts
        ]
        self._plddt_regions = self.get_plddt_regions()
        self._h_score = self.calculate_h_score()
        self.__name = self.cif_file.stem

    @property
    def name(self):
        return self.__name

    @name.setter
    def name(self, name: str):
        if not isinstance(name, str):
            logger.error("Name must be a string")
            raise ValueError()
        self.__name = name

    @property
    def plddts(self):
        """
        The pLDDT scores for each atom in the model
        """
        return self.__plddts
    
    @property
    def residue_plddts(self):
        """
        The pLDDT scores for each residue in the model
        """
        return self.__residue_plddts
    
    @property
    def average_plddt(self):
        """
        The average pLDDT score for the model
        """
        return float(np.mean(self.__plddts))
    
    @property
    def ligand_plddts(self):
        """
        The pLDDT scores for each ligand in the model
        """
        return self.__ligand_plddts
    
    @property
    def h_score(self):
        """
        The H score for the model
        """
        return self._h_score
    
    @property
    def plddt_regions(self):
        """
        The pLDDT regions for the model
        """
        return self._plddt_regions

    def load_cif_file(self):
        """
        Load the cif file using BioPython
        """
        parser = MMCIFParser(QUIET=True)
        return parser.get_structure(self.pathway.stem, self.pathway)

    def chain_lengths(self, mode=ModelCount.RESIDUES, ligand_atoms=False) -> dict:
        """
        Function to get the length of each chain in the model

        Args:
            mode (ModelCount): Enum class specifying the mode to use
            Note: For ligands the length will always be the number of atoms

        Returns:
            dict: Dictionary containing the chain id and the length of the chain

        Raises:
            ValueError: If the mode is not valid
        """
        chains = self.model[0]
        if mode == ModelCount.ALL or mode == ModelCount.ALL.value:

            return {
                chain.id: len([atom for resiude in chain for atom in resiude])
                for chain in chains
            }

        elif mode == ModelCount.RESIDUES or mode == ModelCount.RESIDUES.value:
            residue_counts = {}
            for chain in chains:
                if self.check_ligand(chain):
                    if ligand_atoms:
                        residue_counts[chain.id] = len(
                            [atom for resiude in chain for atom in resiude]
                        )
                    else:
                        residue_counts[chain.id] = 1
                    continue
                residue_counts[chain.id] = len(chain)
            return residue_counts

        else:
            msg = f"Invalid mode. Please use {', '.join(ModelCount.__members__)}"
            logger.critical(msg)
            raise ValueError()

    def token_residue_ids(self) -> dict:
        """
        Function to get the residue ids for each chain in the model

        Returns:
            dict: Dictionary containing the chain id and the residue ids for each
            chain
        """
        chains = self.model[0]
        residue_ids = {}
        for chain in chains:
            if self.check_ligand(chain):
                residue_ids[chain.id] = [
                    residue.id[1] for residue in chain for atom in residue
                ]
                continue
            residue_ids[chain.id] = [
                residue.id[1]
                for residue in chain
                if residue.id[0] == " " or residue.id[0] == "H"
            ]

        return residue_ids
    
    def calculate_h_score(self):
        """
        Calculate the H score for the model

        Returns:
            float: The H score for the model
        """
       
        score = 0
        for i in reversed(range(1, 101)):
            if (100.0 / len(self.plddts)) * np.sum(np.array(self.plddts) >= i) >= i:
                score = i
                break
        return score
    
    def get_model_sequence_data(self) -> dict:
        """
        Get the sequence for each chain and ligand in the model, used internally for plotting

        Returns:
            dict : Chain ID and sequence data
        """
        sequence_data = {}
        for chain in self.model[0]:
            if self.check_ligand(chain):
                sequence_data[chain.id] = "".join([atom.id[0] for residue in chain for atom in residue])
            else:
                sequence_data[chain.id] = "".join([seq1(residue.get_resname()) for residue in chain])
        return sequence_data

    def get_plddt_per_atom(self) -> dict:
        """
        Get the pLDDT scores for each atom in the model

        Returns:
            dict: Dictionary containing the chain id and the pLDDT scores for each atom
        """
        plddt: Dict[str, list] = {}
        for chain in self.model[0]:
            if self.check_ligand(chain):
                continue
            for residue in chain:
                for atom in residue:
                    if chain.id in plddt:
                        plddt[chain.id].append(atom.bfactor)
                    else:
                        plddt[chain.id] = [atom.bfactor]

        return plddt

    def get_plddt_per_residue(self, method=ResidueCountType.AVERAGE.value) -> dict:
        """
        Get the pLDDT scores for each residue in the model

        Args:
            method (ResidueCountType): Enum class specifying the method to use

        Returns:
            dict: Dictionary containing the chain id and the pLDDT scores for each
            residue
        """
        plddts: Dict[str, list] = {}

        if method not in ResidueCountType.values():
            logger.error(
                f"Invalid method. Please use {', '.join(ResidueCountType.__members__)}"
            )
            raise ValueError()

        for chain in self.model[0]:
            # if self.check_ligand(chain):
            #     continue
            for residue in chain:
                if method == ResidueCountType.AVERAGE.value:
                    scores = 0
                    for atom in residue:
                        scores += atom.bfactor
                    score = scores / len(residue)

                elif method == ResidueCountType.CARBONALPHA.value:
                    for atom in residue:
                        if atom.id == "CA":
                            score = atom.bfactor
                            break

                if chain.id in plddts:
                    plddts[chain.id].append(score)

                else:
                    plddts[chain.id] = [score]

        return plddts
    
    def get_plddt_per_ligand(self) -> dict:
        """
        Get the pLDDT scores for each ligand in the model

        Returns:
            dict: Dictionary containing the chain id and the pLDDT scores for each atom
        """
        plddt: Dict[str, list] = {}
        for chain in self.model[0]:
            if self.check_ligand(chain):
                for residue in chain:
                    for atom in residue:
                        if chain.id in plddt:
                            plddt[chain.id].append(atom.bfactor)
                        else:
                            plddt[chain.id] = [atom.bfactor]
        return plddt
    
    def get_plddt_regions(self) -> dict:
        """
        Get the pLDDT regions for the model 
        """
        regions = {}

        plddts_array = np.array(self.residue_plddts + self.ligand_plddts)
        v_low = np.where(plddts_array <= 50)[0]
        regions['v_low'] = self._get_regions(v_low)
        low = np.where((plddts_array > 50) & (plddts_array < 70))[0]
        regions['low'] = self._get_regions(low)
        confident = np.where((plddts_array >= 70) & (plddts_array < 90))[0]
        regions['confident'] = self._get_regions(confident)
        v_confident = np.where(plddts_array >= 90)[0]
        regions['v_high'] = self._get_regions(v_confident)

        return regions
    
    def _get_regions(self, indices):
        """
        Get the regions from the indices
        """
        regions = []
        for _, g in groupby(enumerate(indices), lambda x: x[0] - x[1]):
            group = (map(itemgetter(1), g))
            group = list(map(int, group))
            regions.append((group[0], group[-1]))
        return regions


    def check_ligand(self, chain: Chain) -> bool:
        """
        Check if the chain is a ligand

        Args:
            chain (Chain): BioPython chain object

        Returns:
            bool: True if the chain is a ligand, False otherwise
        """

        sequences = self.input_params.get("sequences")
        if sequences is None:
            return False
        for sequence in sequences:
            for sequence_type, sequence_data in sequence.items():
                if sequence_type == "ligand":
                    if "id" not in sequence_data:
                        continue
                    if isinstance(sequence_data["id"], str):
                        if chain.id == sequence_data["id"]:
                            return True
                    elif isinstance(sequence_data["id"], list):
                        if chain.id in sequence_data["id"]:
                            return True
        return False
    
    def to_file(self, output_file: Union[str, Path]) -> None:
        """
        Save the cif file

        Args:
            output_file (Union[str, Path]): Path to save the cif file

        Returns:
            None
        """
        io = MMCIFIO()
        io.set_structure(self.model)
        io.save(str(output_file))

    def check_clashes(
        self, threshold: Union[int, float] = 2.4, bucket: int = 10
    ) -> List[Tuple[Atom, Atom]]:
        """
        Check for clashes between atoms in different chains

        Args:
            threshold: The distance threshold for a clash.

        Returns:
            A list of clashes.

        """
        atoms = [atom for chain in self.model[0] for atom in chain.get_atoms()]
        coords = np.array(
            [atom.get_coord() for atom in atoms],
            dtype="d",
        )
        assert bucket > 1
        assert coords.shape[1] == 3

        tree = KDTree(coords, bucket)
        neighbors = tree.neighbor_search(threshold)
        clashes = []

        for neighbor in neighbors:
            i1, i2 = neighbor.index1, neighbor.index2
            atom1, atom2 = atoms[i1], atoms[i2]
            # find chain_id and residue_id
            chain_id1 = atom1.get_full_id()[2]
            chain_id2 = atom2.get_full_id()[2]
            if chain_id1 == chain_id2:
                continue
            else:
                clashes.append((atom1, atom2))
        self.clashes = len(clashes)

        return clashes


class ConfidenceJsonFile(FileBase):
    def __init__(self, json_file: Union[str, Path]):
        """
        Object to handle json files

        Args:
            json_file (Union[str, Path]): Path to the json file

        Attributes:
            json_file (Path): Path to the json file
            data (dict): Dictionary containing the data from the json file

        """
        super().__init__(json_file)
        self.data = self.load_json_file()

    def load_json_file(self):
        # load the json file
        with open(self.pathway, "r") as f:
            data = json.load(f)

        return data
