import json
import logging
import re
import warnings
from abc import ABC
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from Bio.PDB import MMCIFIO, Chain, MMCIFParser, Model
from Bio.PDB.Atom import Atom
from Bio.PDB.kdtrees import KDTree
from Bio.PDB.Polypeptide import is_aa
from Bio.PDB.Residue import Residue
from Bio.SeqUtils import seq1

from abcfold.processoutput.atoms import VANDERWALLS

warnings.filterwarnings("ignore")

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
        self.clashes_residues = 0
        self.model = self.load_cif_file()
        self.__ligand_plddts = None
        self.__plddts = None
        self.__residue_plddts = None
        self.__h_score = None
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
        self.__plddts = [
            plddts for plddts in self.get_plddt_per_atom().values() for plddts in plddts
        ]
        return self.__plddts

    @property
    def residue_plddts(self):
        """
        The pLDDT scores for each residue in the model
        """
        self.__residue_plddts = [
            plddts
            for plddts in self.get_plddt_per_residue().values()
            for plddts in plddts
        ]
        return self.__residue_plddts

    @property
    def average_plddt(self):
        """
        The average pLDDT score for the model
        """
        return float(np.mean(self.plddts))

    @property
    def ligand_plddts(self):
        """
        The pLDDT scores for each ligand in the model
        """
        self.__ligand_plddts = self.get_plddt_per_ligand()
        return self.__ligand_plddts

    @property
    def h_score(self):
        """
        The H score for the model
        """
        self.__h_score = self.calculate_h_score()
        return self.__h_score

    def load_cif_file(self):
        """
        Load the cif file using BioPython
        """
        parser = MMCIFParser(QUIET=True)
        return parser.get_structure(self.pathway.stem, self.pathway)

    def get_chains(self):
        return self.model[0]

    def chain_lengths(
        self,
        mode=ModelCount.RESIDUES,
        ligand_atoms=False,
        ptm_atoms=False,
    ) -> dict:
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
        chains = self.get_chains()
        if mode == ModelCount.ALL or mode == ModelCount.ALL.value:

            # return {
            #     chain.id: len([atom for resiude in chain for atom in resiude])
            #     for chain in chains
            # }
            chain_lengths: dict = {}
            for chain in chains:
                if chain.id in chain_lengths:
                    chain_lengths[chain.id] += len(
                        [atom for resiude in chain for atom in resiude]
                    )
                else:
                    chain_lengths[chain.id] = len(
                        [atom for resiude in chain for atom in resiude]
                    )

            return chain_lengths

        elif mode == ModelCount.RESIDUES or mode == ModelCount.RESIDUES.value:
            residue_counts: dict = {}
            for chain in chains:
                if self.check_other(chain, ["protein", "rna", "dna"]) and ptm_atoms:
                    counter = 0
                    for residue in chain:
                        if is_aa(residue.resname, standard=True):
                            counter += 1
                            continue
                        else:
                            counter += len([atom for atom in residue])

                    residue_counts[chain.id] = counter

                elif self.check_ligand(chain):
                    if ligand_atoms:
                        if chain.id in residue_counts:
                            residue_counts[chain.id] += len(
                                [atom for resiude in chain for atom in resiude]
                            )
                        else:
                            residue_counts[chain.id] = len(
                                [atom for resiude in chain for atom in resiude]
                            )

                    else:
                        residue_counts[chain.id] = 1
                else:
                    residue_counts[chain.id] = len([residue for residue in chain])

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
        from abcfold.processoutput.utils import flatten

        chains = self.get_chains()
        residue_ids = {}
        for chain in chains:
            if self.check_ligand(chain):
                residue_ids[chain.id] = [
                    [residue.id[1]] for residue in chain for _ in residue
                ]
                continue
            residue_ids[chain.id] = [
                (
                    [residue.id[1]]
                    if residue.id[0] == " " or residue.id[0] == "H"
                    else [residue.id[1] for _ in residue]
                )
                for residue in chain
            ]

        residue_ids = {k: flatten(v) for k, v in residue_ids.items()}

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
        Get the sequence for each chain and ligand in the model, used internally
        for plotting

        Returns:
            dict : Chain ID and sequence data
        """
        sequence_data = {}
        for chain in self.model[0]:
            if self.check_ligand(chain):
                sequence_data[chain.id] = "".join(
                    [atom.id[0] for residue in chain for atom in residue]
                )
            else:
                sequence_data[chain.id] = "".join(
                    [seq1(residue.get_resname()) for residue in chain]
                )
        return sequence_data

    def get_plddt_per_atom(self) -> dict:
        """
        Get the pLDDT scores for each atom in the model

        Returns:
            dict: Dictionary containing the chain id and the pLDDT scores for each atom
        """
        plddt: Dict[str, list] = {}
        for chain in self.model[0]:

            for residue in chain:
                for atom in residue:
                    if chain.id in plddt:
                        plddt[chain.id].append(atom.bfactor)
                    else:
                        plddt[chain.id] = [atom.bfactor]

        return plddt

    def get_plddt_per_residue(self, method=ResidueCountType.CARBONALPHA.value) -> dict:
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

        chains = self.get_chains()
        for chain in chains:
            if self.check_ligand(chain):
                if chain.id in plddts:
                    plddts[chain.id].extend(
                        [atom.bfactor for residue in chain for atom in residue]
                    )
                else:
                    plddts[chain.id] = [
                        atom.bfactor for residue in chain for atom in residue
                    ]

            else:
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

        plddt_lengths = {k: len(v) for (k, v) in plddts.items()}
        chain_lengths = self.chain_lengths(mode="residues", ligand_atoms=True)
        for chain_id in plddt_lengths:
            assert (
                chain_lengths[chain_id] == plddt_lengths[chain_id]
            ), f"{chain_id}, {chain_lengths[chain_id]} != {plddt_lengths[chain_id]}"
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

    def check_ligand(self, chain: Chain) -> bool:
        """
        Check if the chain is a ligand

        Args:
            chain (Chain): BioPython chain object

        Returns:
            bool: True if the chain is a ligand, False otherwise
        """

        return self.check_other(chain, ["ligand"])

    def check_other(self, chain: Chain, check_list) -> bool:
        sequences = self.input_params.get("sequences")
        if sequences is None:
            logger.warning("Unable to gain sequence infromation from input file")
            return False
        for sequence in sequences:
            for sequence_type, sequence_data in sequence.items():
                if sequence_type in check_list:
                    if "id" not in sequence_data:
                        continue
                    if hasattr(chain, "id"):
                        chain_id = chain.id
                    else:
                        chain_id = chain
                    if isinstance(sequence_data["id"], str):
                        if chain_id == sequence_data["id"]:
                            return True
                    elif isinstance(sequence_data["id"], list):
                        if chain_id in sequence_data["id"]:
                            return True
        return False

    def relabel_chains(
        self, chain_ids: List[str], link_ids: Optional[dict] = None
    ) -> None:
        """
        Relabel the chains in the model

        Args:
            chain_ids (List[str]): List of chain ids to relabel the chains

        Returns:
            None
        """

        chain_ids = chain_ids.copy()
        structure = self.model[0]
        old_new_chain_id = {}

        if link_ids is None:
            link_ids = {}
        else:
            for new_ids in link_ids.values():
                for new_id in new_ids:
                    chain_ids.pop(chain_ids.index(new_id))

        old_chain_label_counter, new_chain_label_counter = 0, 0

        chain_names = [chain.id for chain in structure]
        while old_chain_label_counter < len(structure):
            chain = chain_ids[new_chain_label_counter]
            old_new_chain_id[chain_names[old_chain_label_counter]] = chain

            # increment the old_chain everytime a chain has been relabelled
            old_chain_label_counter += 1

            if chain in link_ids:
                ligand_no_added = 2
                for _ in link_ids[chain]:
                    old_new_chain_id[chain_names[old_chain_label_counter]] = chain
                    for residue in structure[chain_names[old_chain_label_counter]]:

                        residue.id = (residue.id[0], ligand_no_added, residue.id[2])
                        ligand_no_added += 1
                    old_chain_label_counter += 1

            new_chain_label_counter += 1

        for chain_to_rename in structure:
            chain_to_rename.id = old_new_chain_id[chain_to_rename.id]

        assert old_chain_label_counter == len(
            self.get_chains()
        ), "Number of chain ids must match the number of chains"
        self.update()

    def update(self):
        self.to_file(self.pathway)
        self = CifFile(self.pathway, self.input_params)

    def reorder_chains(self, new_chain_ids: List[str]):

        assert sorted([chain.id for chain in self.get_chains()]) == sorted(
            new_chain_ids
        ), "The chain ids need to be identical to what is in the model already \
for reordering"

        new_model = Model.Model(0)

        [new_model.add(self.model[0][ch]) for ch in new_chain_ids]
        self.model.detach_child(0)
        self.model.add(new_model)
        self.update()

    def check_clashes(
        self,
        threshold: Union[int, float] = 3.4,
        bucket: int = 10,
        clash_cutoff: float = 0.63,
    ) -> Tuple[List[Tuple[Atom, Atom]], List[Tuple[Residue, Residue]]]:
        """
        Check for clashes between atoms in different chains

        Args:
            threshold: The distance threshold for a clash.

        Returns:
            A list of clashes.

        """
        atoms = self.get_atoms()
        coords = np.array(
            [atom.get_coord() for atom in atoms],
            dtype="d",
        )
        assert bucket > 1
        assert coords.shape[1] == 3
        assert clash_cutoff > 0.0 and clash_cutoff <= 1.0

        tree = KDTree(coords, bucket)
        neighbors = tree.neighbor_search(threshold)
        clashes_atoms, clashes_residues = [], []

        for neighbor in neighbors:
            i1, i2 = neighbor.index1, neighbor.index2
            atom1, atom2 = atoms[i1], atoms[i2]
            # get the element of the atom
            element1 = atom1.element
            element2 = atom2.element
            # find chain_id and residue_id
            chain_id1 = atom1.get_full_id()[2]
            chain_id2 = atom2.get_full_id()[2]

            if chain_id1 == chain_id2:
                continue

            distance = np.linalg.norm(atom1.get_coord() - atom2.get_coord())
            if (atom1.name == "C" and atom1.name == "N") or (
                atom2.name == "N" and atom1.name == "C"
            ):
                continue
            elif (atom1.name == "SG" and atom2.name == "SG") and distance > 1.88:
                continue

            clash_radius = (
                VANDERWALLS.get(element1, 1.7) + VANDERWALLS.get(element2, 1.7)
            ) * 0.63
            if distance < clash_radius:
                residue1 = atom1.get_parent()
                residue2 = atom2.get_parent()

                clashes_atoms.append((atom1, atom2))

                if (residue1, residue2) not in clashes_residues:
                    clashes_residues.append((residue1, residue2))

        self.clashes = len(clashes_atoms)
        self.clashes_residues = len(clashes_residues)
        return (clashes_atoms, clashes_residues)

    def get_atoms(self, chain_id=None) -> list:
        """
        Get the atoms of the structure

        Args:
            threshold: The distance threshold for a clash.

        Returns:
            A list of clashes.

        """
        if chain_id is not None:
            return [
                atom
                for chain in self.model[0]
                for atom in chain.get_atoms()
                if chain.id == chain_id
            ]
        return [atom for chain in self.model[0] for atom in chain.get_atoms()]

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

        # save creates the dictionary
        io.save(str(output_file))
        self.__atom_site_label_update(io.dic)
        self.__ligand_to_hetatm(io.dic)
        with open(output_file, "w") as f:
            io._save_dict(f)

        self.__single_to_double_quotes(output_file)

    def __single_to_double_quotes(self, file_name: Union[str, Path]) -> None:
        new_lines = []
        with open(file_name, "r") as f:
            lines = [line.rstrip() for line in f]

        for line in lines:

            single_quotes = re.compile(r"[']\w+[']{2}")
            new_lines.append(
                single_quotes.sub(lambda x: f'"{x.group()[1:-1]}"', line, count=1)
            )

        with open(file_name, "w") as f:
            f.write("\n".join(new_lines))

    def __atom_site_label_update(self, out_dict):
        atom_site_labels_asym_ids = []
        for chain_id, chain_length in self.chain_lengths(mode="all").items():
            atom_site_labels_asym_ids.extend([chain_id] * chain_length)

        assert len(out_dict["_atom_site.label_asym_id"]) == len(
            atom_site_labels_asym_ids
        ), f"Lengths must be the same, current lengths are \
{len(out_dict['_atom_site.label_asym_id'])} and {len(atom_site_labels_asym_ids)}"

        out_dict["_atom_site.label_asym_id"] = atom_site_labels_asym_ids

        return out_dict

    def __ligand_to_hetatm(self, out_dict):
        atom_site_group_pdb = []
        counter = 0
        for chain_id, chain_length in self.chain_lengths(mode="all").items():
            if self.check_ligand(chain_id):
                atom_site_group_pdb.extend(["HETATM"] * chain_length)
            else:
                atom_site_group_pdb.extend(
                    out_dict["_atom_site.group_PDB"][counter:counter+chain_length]
                )

            counter += chain_length

        assert len(out_dict["_atom_site.group_PDB"]) == len(atom_site_group_pdb)

        out_dict["_atom_site.group_PDB"] = atom_site_group_pdb

        return out_dict


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
