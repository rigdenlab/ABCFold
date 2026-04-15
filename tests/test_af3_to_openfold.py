import json
import tempfile
from pathlib import Path

from abcfold.openfold3.af3_to_openfold3 import OpenfoldJson

# flake8: noqa

def test_af3_to_openfold(test_data):
    with tempfile.TemporaryDirectory() as temp_dir:
        openfold_json = OpenfoldJson(temp_dir)

        data = openfold_json.json_to_json(test_data.test_inputAB_json)

        reference = {
            "queries":{
                "2PV7": {
                    "chains": [
                        {
                            "molecule_type": "protein",
                            "chain_ids": ["A", "B"],
                            "sequence": "GMRES"
                        },
                        {
                            "molecule_type": "protein",
                            "chain_ids": "C",
                            "sequence": "YANEN"
                        },
                        {
                            "molecule_type": "ligand",
                            "chain_ids": ["D", "E"],
                            "ccd_codes": "ATP"
                        },
                        {
                            "molecule_type": "ligand",
                            "chain_ids": "F",
                            "smiles": "CC(=O)OC1C[NH+]2CCC1CC2"
                        }
                    ]
                }
            }
        }

        assert data == reference


def test_af3_to_openfold_rna(test_data):
    with tempfile.TemporaryDirectory() as temp_dir:
        openfold_json = OpenfoldJson(temp_dir)

        data = openfold_json.json_to_json(test_data.test_inputRNA_json)

        reference = {
            "queries":{
                "RNA_example": {
                    "chains": [
                        {
                            "molecule_type": "rna",
                            "chain_ids": "A",
                            "sequence": "AGCU"
                        },
                    ]
                }
            }
        }

        assert data == reference


def test_af3_to_openfold_dna(test_data):
    with tempfile.TemporaryDirectory() as temp_dir:
        openfold_json = OpenfoldJson(temp_dir)

        data = openfold_json.json_to_json(test_data.test_inputDNA_json)

        reference = {
            "queries":{
                "DNA_example": {
                    "chains": [
                        {
                            "molecule_type": "dna",
                            "chain_ids": ["A", "B"],
                            "sequence": "AGCT"
                        },
                    ]
                }
            }
        }

        assert data == reference


def test_af3_to_openfold_ligand(test_data):
    with tempfile.TemporaryDirectory() as temp_dir:
        openfold_json = OpenfoldJson(temp_dir)

        data = openfold_json.json_to_json(test_data.test_inputLIG_json)

        reference = {
            "queries":{
                "2PV7": {
                    "chains": [
                        {
                            "molecule_type": "protein",
                            "chain_ids": ["A", "B"],
                            "sequence": "GMRESYANENQFGFKTINSDIHKIVIVGGYGKLGGLFARYLRASGYPISILDREDWAVAESILANADVVIVSVPINLTLETIERLKPYLTENMLLADLTSVKREPLAKMLEVHTGAVLGLHPMFGADIASMAKQVVVRCDGRFPERYEWLLEQIQIWGAKIYQTNATEHDHNMTYIQALRHFSTFANGLHLSKQPINLANLLALSSPIYRLELAMIGRLFAQDAELYADIIMDKSENLAVIETLKQTYDEALTFFENNDRQGFIDAFHKVRDWFGDYSEQFLKESRQLLQQANDLKQG"
                        },
                        {
                            "molecule_type": "ligand",
                            "chain_ids": ["C", "D"],
                            "ccd_codes": "ATP"
                        },
                        {
                            "molecule_type": "ligand",
                            "chain_ids": "E",
                            "smiles": "CC(=O)OC1C[NH+]2CCC1CC2"
                        },
                        {
                            "molecule_type": "ligand",
                            "chain_ids": ["G", "H"],
                            "smiles": "CCCCCCCCCCCC(O)=O"
                        },
                        {
                            "molecule_type": "ligand",
                            "chain_ids": "F",
                            "ccd_codes": "MG"
                        }
                    ]
                }
            }
        }

        assert data == reference


def test_af3_to_openfold_ptm(test_data):
    with tempfile.TemporaryDirectory() as temp_dir:
        openfold_json = OpenfoldJson(temp_dir)

        data = openfold_json.json_to_json(test_data.test_inputPTM_json)

        reference = {
            "queries":{
                "PTM example": {
                    "chains": [
                        {
                            "molecule_type": "protein",
                            "chain_ids": "A",
                            "sequence": "PVLSCGEWQL",
                            "non_canonical_residues": {
                                    "1": "HY3",
                                    "5": "P1L"
                            }
                        },
                        {
                            "molecule_type": "rna",
                            "chain_ids": "B",
                            "sequence": "AGCU",
                            "non_canonical_residues": {
                                    "1": "2MG",
                                    "4": "5MC"
                            }
                        }
                    ]
                }
            }
        }

        assert data == reference


def test_openfold_output_msa(test_data):
    with tempfile.TemporaryDirectory() as temp_dir:
        openfold_json = OpenfoldJson(temp_dir)

        data = openfold_json.json_to_json(test_data.test_inputAmsa_json)
        msa_path = data["queries"]["2PV7"]["chains"][0]["main_msa_file_paths"]

        # MSA directory has a random path, so just check that it exists then give
        # it a placeholder value for comparison
        assert msa_path is not None
        assert Path(msa_path[0]).exists()
        data["queries"]["2PV7"]["chains"][0]["main_msa_file_paths"] = (
            ["MSA_DIR"]
        )

        reference = {
            "queries":{
                "2PV7": {
                    "use_msas": True,
                    "use_main_msas": True,
                    "use_paired_msas": False,
                    "chains": [
                        {
                            "molecule_type": "protein",
                            "chain_ids": "A",
                            "sequence": "GMRESYANENQFGFKTINSDIHKIVIVGGYGKLGGLFARYLRASGYPISILDREDWAVAESILANADVVIVSVPINLTLETIERLKPYLTENMLLADLTSVKREPLAKMLEVHTGAVLGLHPMFGADIASMAKQVVVRCDGRFPERYEWLLEQIQIWGAKIYQTNATEHDHNMTYIQALRHFSTFANGLHLSKQPINLANLLALSSPIYRLELAMIGRLFAQDAELYADIIMDKSENLAVIETLKQTYDEALTFFENNDRQGFIDAFHKVRDWFGDYSEQFLKESRQLLQQANDLKQG",
                            "main_msa_file_paths": ["MSA_DIR"],
                        },
                    ]
                }
            }
        }

        assert data == reference


def test_openfold_write_json(test_data):
    with tempfile.TemporaryDirectory() as temp_dir:
        openfold_json = OpenfoldJson(temp_dir)

        openfold_json.json_to_json(test_data.test_inputAB_json)

        out_file = Path(temp_dir) / "openfold_output.json"
        openfold_json.write_json(out_file)

        reference = {
            "queries":{
                "2PV7": {
                    "chains": [
                        {
                            "molecule_type": "protein",
                            "chain_ids": ["A", "B"],
                            "sequence": "GMRES"
                        },
                        {
                            "molecule_type": "protein",
                            "chain_ids": "C",
                            "sequence": "YANEN"
                        },
                        {
                            "molecule_type": "ligand",
                            "chain_ids": ["D", "E"],
                            "ccd_codes": "ATP"
                        },
                        {
                            "molecule_type": "ligand",
                            "chain_ids": "F",
                            "smiles": "CC(=O)OC1C[NH+]2CCC1CC2"
                        }
                    ]
                }
            }
        }

        with open(out_file, "r") as f:
            written_data = f.read()
        written_data = json.loads(written_data)

        assert written_data == reference

def test_openfold_write_yaml(test_data):
    with tempfile.TemporaryDirectory() as temp_dir:
        openfold_json = OpenfoldJson(temp_dir)

        openfold_json.json_to_json(test_data.test_inputAB_json)

        out_file = Path(temp_dir) / "openfold_output.yml"
        yaml_string = openfold_json.write_yaml(out_file)
        yaml_string = yaml_string.split("\n")

        assert yaml_string[0] == "model_update:"
        assert yaml_string[1] == "  presets:"
        assert yaml_string[2] == "    - predict"
        assert yaml_string[3] == "    - low_mem"
        assert yaml_string[4] == "  custom:"
        assert yaml_string[5] == "    settings:"
        assert yaml_string[6] == "      memory:"
        assert yaml_string[7] == "        eval:"
        assert yaml_string[8] == "          use_cueq_triangle_kernels: true"
        assert yaml_string[9] == "          use_deepspeed_evo_attention: false"
        assert yaml_string[10] == "experiment_settings:"
        assert yaml_string[11] == "  mode: predict"
        assert yaml_string[12] == "  seeds: [1]"
        assert yaml_string[13] == "  use_templates: false"
