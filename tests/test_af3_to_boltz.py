import json
import tempfile
from pathlib import Path

from abcfold.boltz1.af3_to_boltz1 import DELIM, BoltzYaml


def test_af3_to_boltz(test_data):
    with tempfile.TemporaryDirectory() as temp_dir:
        boltz_yaml = BoltzYaml(temp_dir)

        yaml_string_multi = boltz_yaml.json_to_yaml(test_data.test_inputAB_json)

        yaml_string_multi = yaml_string_multi.split("\n")

        assert yaml_string_multi[0] == "version: 1"
        assert yaml_string_multi[1] == "sequences:"
        assert yaml_string_multi[2] == f"{DELIM}- protein:"
        assert yaml_string_multi[3] == f"{DELIM}{DELIM}id: [A, B]"
        assert yaml_string_multi[4] == f"{DELIM}{DELIM}sequence: GMRES"
        assert yaml_string_multi[5] == f"{DELIM}- protein:"
        assert yaml_string_multi[6] == f"{DELIM}{DELIM}id: C"
        assert yaml_string_multi[7] == f"{DELIM}{DELIM}sequence: YANEN"
        assert yaml_string_multi[8] == f"{DELIM}- ligand:"
        assert yaml_string_multi[9] == f"{DELIM}{DELIM}id: [D, E]"
        assert yaml_string_multi[10] == f"{DELIM}{DELIM}ccd: ATP"
        assert yaml_string_multi[11] == f"{DELIM}- ligand:"
        assert yaml_string_multi[12] == f"{DELIM}{DELIM}id: F"
        assert yaml_string_multi[13] == f"{DELIM}{DELIM}smiles: CC(=O)OC1C[NH+]2CCC1CC2"

        yaml_string_bonds = boltz_yaml.json_to_yaml(test_data.test_inputA_json)
        yaml_string_bonds = yaml_string_bonds.split("\n")

        print(yaml_string_bonds)
        assert yaml_string_bonds[0] == "version: 1"
        assert yaml_string_bonds[1] == "sequences:"
        assert yaml_string_bonds[2] == f"{DELIM}- protein:"
        assert yaml_string_bonds[3] == f"{DELIM}{DELIM}id: A"
        assert (
            yaml_string_bonds[4]
            == f"{DELIM}{DELIM}sequence: GMRESYANENQFGFKTINSDIHKIVIVGGYGKLGGLFARY\
LRASGYPISILDREDWAVAESILANADVVIVSVPINLTLETIERLKPYLTENMLLADLTSVKREPLAKMLEVHTGAVLGLHPMFG\
ADIASMAKQVVVRCDGRFPERYEWLLEQIQIWGAKIYQTNATEHDHNMTYIQALRHFSTFANGLHLSKQPINLANLLALSSPIYRL\
ELAMIGRLFAQDAELYADIIMDKSENLAVIETLKQTYDEALTFFENNDRQGFIDAFHKVRDWFGDYSEQFLKESRQLLQQANDLKQG"
        )

        with open(test_data.test_inputAB_json, "r") as f:
            json_dict = json.load(f)

        yaml_string = boltz_yaml.json_to_yaml(json_dict)
        yaml_string = yaml_string.split("\n")

        assert yaml_string[0] == "version: 1"
        assert yaml_string[1] == "sequences:"
        assert yaml_string_multi[2] == f"{DELIM}- protein:"
        assert yaml_string_multi[3] == f"{DELIM}{DELIM}id: [A, B]"
        assert yaml_string_multi[4] == f"{DELIM}{DELIM}sequence: GMRES"
        assert yaml_string_multi[5] == f"{DELIM}- protein:"
        assert yaml_string_multi[6] == f"{DELIM}{DELIM}id: C"
        assert yaml_string_multi[7] == f"{DELIM}{DELIM}sequence: YANEN"
        assert yaml_string_multi[8] == f"{DELIM}- ligand:"
        assert yaml_string_multi[9] == f"{DELIM}{DELIM}id: [D, E]"
        assert yaml_string_multi[10] == f"{DELIM}{DELIM}ccd: ATP"
        assert yaml_string_multi[11] == f"{DELIM}- ligand:"
        assert yaml_string_multi[12] == f"{DELIM}{DELIM}id: F"
        assert yaml_string_multi[13] == f"{DELIM}{DELIM}smiles: CC(=O)OC1C[NH+]2CCC1CC2"


def test_boltz_output_yaml(test_data):
    with tempfile.TemporaryDirectory() as temp_dir:
        boltz_yaml = BoltzYaml(temp_dir)

        assert not boltz_yaml.yaml_string
        boltz_yaml.json_to_yaml(test_data.test_inputAB_json)

        assert boltz_yaml.yaml_string

        file_name = Path(temp_dir) / "output.yaml"
        boltz_yaml.write_yaml(file_name)

        assert file_name.exists()
        assert file_name.is_file()


def test_boltz_output_msa(test_data):
    with tempfile.TemporaryDirectory() as temp_dir:
        boltz_yaml = BoltzYaml(temp_dir)

        assert boltz_yaml.msa_file == "null"

        boltz_yaml.json_to_yaml(test_data.test_inputAmsa_json)

        assert boltz_yaml.msa_file != "null"
        assert boltz_yaml.msa_file.exists()
        assert boltz_yaml.msa_file.is_file()


def test_af3_data_json_to_yaml(output_objs):
    try:
        af3_json = output_objs.af3_output.input_json
        with tempfile.TemporaryDirectory() as temp_dir:
            boltz_yaml = BoltzYaml(temp_dir)
            boltz_yaml.json_to_yaml(af3_json)
    except TypeError:
        assert False


def test_extra_ligand_ids():
    difficult_params = {
        "name": "Nightmare",
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
        "dialect": "scouse",
        "version": 2,
    }

    with tempfile.TemporaryDirectory() as temp_dir:
        boltz_yaml = BoltzYaml(temp_dir)
        yaml_string = boltz_yaml.json_to_yaml(difficult_params)
        yaml_string = yaml_string.split("\n")

        assert yaml_string[0] == "version: 1"
        assert yaml_string[1] == "sequences:"
        assert yaml_string[2] == f"{DELIM}- protein:"
        assert yaml_string[3] == f"{DELIM}{DELIM}id: A"
        assert yaml_string[4] == f"{DELIM}{DELIM}sequence: PVLSCGEWQL"
        assert yaml_string[5] == f"{DELIM}{DELIM}modifications:"
        assert yaml_string[6] == f"{DELIM}{DELIM}{DELIM}- position: 1"
        assert yaml_string[7] == f"{DELIM}{DELIM}{DELIM}  ccd: HY3"
        assert yaml_string[8] == f"{DELIM}{DELIM}{DELIM}- position: 5"
        assert yaml_string[9] == f"{DELIM}{DELIM}{DELIM}  ccd: P1L"
        assert yaml_string[10] == f"{DELIM}- protein:"
        assert yaml_string[11] == f"{DELIM}{DELIM}id: B"
        assert yaml_string[12] == f"{DELIM}{DELIM}sequence: QIQLVQSGPELKKPGET"
        assert yaml_string[13] == f"{DELIM}- protein:"
        assert yaml_string[14] == f"{DELIM}{DELIM}id: C"
        assert yaml_string[15] == f"{DELIM}{DELIM}sequence: DVLMIQTPLSLPVS"
        assert yaml_string[16] == f"{DELIM}- ligand:"
        assert yaml_string[17] == f"{DELIM}{DELIM}id: [F]"
        assert yaml_string[18] == f"{DELIM}{DELIM}ccd: ATP"
        assert yaml_string[19] == f"{DELIM}- ligand:"
        assert yaml_string[20] == f"{DELIM}{DELIM}id: I"
        assert yaml_string[21] == f"{DELIM}{DELIM}ccd: NAG"
        assert yaml_string[22] == f"{DELIM}- ligand:"
        assert yaml_string[23] == f"{DELIM}{DELIM}id: E"
        assert yaml_string[24] == f"{DELIM}{DELIM}ccd: FUC"
        assert yaml_string[25] == f"{DELIM}- ligand:"
        assert yaml_string[26] == f"{DELIM}{DELIM}id: G"
        assert yaml_string[27] == f"{DELIM}{DELIM}ccd: FUC"
        assert yaml_string[28] == f"{DELIM}- dna:"
        assert yaml_string[29] == f"{DELIM}{DELIM}id: [D, K]"
        assert yaml_string[30] == f"{DELIM}{DELIM}sequence: AGCT"
        assert yaml_string[31] == f"{DELIM}- rna:"
        assert yaml_string[32] == f"{DELIM}{DELIM}id: L"
        assert yaml_string[33] == f"{DELIM}{DELIM}sequence: AGCU"
        assert yaml_string[34] == f"{DELIM}- ligand:"
        assert yaml_string[35] == f"{DELIM}{DELIM}id: Z"
        assert yaml_string[36] == f"{DELIM}{DELIM}smiles: CC(=O)OC1C[NH+]2CCC1CC2"
        assert yaml_string[37] == "constraints:"
        assert yaml_string[38] == f"{DELIM}- bond:"
        assert yaml_string[39] == f"{DELIM}{DELIM}atom1: ['A', 1, 'CA']"
        assert yaml_string[40] == f"{DELIM}{DELIM}atom2: ['B', 1, 'CA']"
        assert yaml_string[41] == f"{DELIM}- bond:"
        assert yaml_string[42] == f"{DELIM}{DELIM}atom1: ['C', 7, 'CA']"
        assert yaml_string[43] == f"{DELIM}{DELIM}atom2: ['A', 10, 'CA']"
        assert yaml_string[44] == f"{DELIM}- bond:"
        assert yaml_string[45] == f"{DELIM}{DELIM}atom1: ['I', 1, 'O3']"
        assert yaml_string[46] == f"{DELIM}{DELIM}atom2: ['E', 1, 'C1']"
        assert yaml_string[47] == f"{DELIM}- bond:"
        assert yaml_string[48] == f"{DELIM}{DELIM}atom1: ['E', 1, 'C1']"
        assert yaml_string[49] == f"{DELIM}{DELIM}atom2: ['G', 1, 'C1']"
