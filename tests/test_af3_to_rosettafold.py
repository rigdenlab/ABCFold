import json
import tempfile
from pathlib import Path

from abcfold.rosettafold3.af3_to_rosettafold3 import Rosettafoldjson

# flake8: noqa


def test_af3_to_rosettafold(test_data):
    with tempfile.TemporaryDirectory() as temp_dir:
        rosettafold_json = Rosettafoldjson(temp_dir)

        data = rosettafold_json.json_to_json(test_data.test_inputAB_json)

        reference = {
            'name': '2PV7',
            'components':
                [
                    {'seq': 'GMRES', 'chain_ids': 'A'},
                    {'seq': 'GMRES', 'chain_ids': 'B'},
                    {'seq': 'YANEN', 'chain_ids': 'C'},
                    {'ccd_code': 'ATP'},
                    {'ccd_code': 'ATP'},
                    {'smiles': 'CC(=O)OC1C[NH+]2CCC1CC2'}
                ]
        }


        assert data == reference


def test_af3_to_rosettafold_rna(test_data):
    with tempfile.TemporaryDirectory() as temp_dir:
        rosettafold_json = Rosettafoldjson(temp_dir)

        data = rosettafold_json.json_to_json(test_data.test_inputRNA_json)
        reference = {
            'name': 'RNA_example',
            'components':
                [
                    {'seq': 'AGCU', 'chain_ids': 'A'},
                ]
        }

        assert data == reference


def test_af3_to_rosettafold_dna(test_data):
    with tempfile.TemporaryDirectory() as temp_dir:
        rosettafold_json = Rosettafoldjson(temp_dir)

        data = rosettafold_json.json_to_json(test_data.test_inputDNA_json)
        reference = {
            'name': 'DNA_example',
            'components':
                [
                    {'seq': 'AGCT', 'chain_ids': 'A'},
                    {'seq': 'AGCT', 'chain_ids': 'B'}
                ]
        }

        assert data == reference


def test_af3_to_rosettafold_ligand(test_data):
    with tempfile.TemporaryDirectory() as temp_dir:
        rosettafold_json = Rosettafoldjson(temp_dir)

        data = rosettafold_json.json_to_json(test_data.test_inputLIG_json)
        reference = {
            'name': '2PV7',
            'components':
                [
                    {'seq': 'GMRESYANENQFGFKTINSDIHKIVIVGGYGKLGGLFARYLRASGYPISILDREDWAVAESILANADVVIVSVPINLTLETIERLKPYLTENMLLADLTSVKREPLAKMLEVHTGAVLGLHPMFGADIASMAKQVVVRCDGRFPERYEWLLEQIQIWGAKIYQTNATEHDHNMTYIQALRHFSTFANGLHLSKQPINLANLLALSSPIYRLELAMIGRLFAQDAELYADIIMDKSENLAVIETLKQTYDEALTFFENNDRQGFIDAFHKVRDWFGDYSEQFLKESRQLLQQANDLKQG', 'chain_ids': 'A'},
                    {'seq': 'GMRESYANENQFGFKTINSDIHKIVIVGGYGKLGGLFARYLRASGYPISILDREDWAVAESILANADVVIVSVPINLTLETIERLKPYLTENMLLADLTSVKREPLAKMLEVHTGAVLGLHPMFGADIASMAKQVVVRCDGRFPERYEWLLEQIQIWGAKIYQTNATEHDHNMTYIQALRHFSTFANGLHLSKQPINLANLLALSSPIYRLELAMIGRLFAQDAELYADIIMDKSENLAVIETLKQTYDEALTFFENNDRQGFIDAFHKVRDWFGDYSEQFLKESRQLLQQANDLKQG', 'chain_ids': 'B'},
                    {'ccd_code': 'ATP'},
                    {'ccd_code': 'ATP'},
                    {'smiles': 'CC(=O)OC1C[NH+]2CCC1CC2'},
                    {'smiles': 'CCCCCCCCCCCC(O)=O'},
                    {'smiles': 'CCCCCCCCCCCC(O)=O'},
                    {'ccd_code': 'MG'}
                ]
        }

        assert data == reference


def test_af3_to_rosettafold_ptm(test_data):
    with tempfile.TemporaryDirectory() as temp_dir:
        rosettafold_json = Rosettafoldjson(temp_dir)

        data = rosettafold_json.json_to_json(test_data.test_inputPTM_json)
        reference = {
            'name': 'PTM example',
            'components':
                [
                    {'seq': '(HY3)VLS(P1L)GEWQL', 'chain_ids': 'A'},
                    {'seq': '(2MG)GC(5MC)', 'chain_ids': 'B'}
                ]
        }

        assert data == reference

def test_rosettafold_output_msa(test_data):
    with tempfile.TemporaryDirectory() as temp_dir:
        rosettafold_json = Rosettafoldjson(temp_dir)

        data = rosettafold_json.json_to_json(test_data.test_inputAmsa_json)
        msa_path = (
            data[0]["components"][0].get("msa_path")[0]
        )
        # MSA directory has a random path, so just check that it exists then give
        # it a placeholder value for comparison
        assert msa_path is not None
        assert Path(msa_path).exists()
        data[0]["components"][0]["msa_path"] = (
            "PRECOMPUTED_MSA_DIR"
        )

        reference = {
            'name': '2PV7',
            'components':
                [
                    {'seq': 'GMRESYANENQFGFKTINSDIHKIVIVGGYGKLGGLFARYLRASGYPISILDREDWAVAESILANADVVIVSVPINLTLETIERLKPYLTENMLLADLTSVKREPLAKMLEVHTGAVLGLHPMFGADIASMAKQVVVRCDGRFPERYEWLLEQIQIWGAKIYQTNATEHDHNMTYIQALRHFSTFANGLHLSKQPINLANLLALSSPIYRLELAMIGRLFAQDAELYADIIMDKSENLAVIETLKQTYDEALTFFENNDRQGFIDAFHKVRDWFGDYSEQFLKESRQLLQQANDLKQG',
                     'chain_ids': 'A',
                     'msa_path': 'PRECOMPUTED_MSA_DIR'},
                ]
        }

        assert data == reference

def test_rosettafold_write_json(test_data):
    with tempfile.TemporaryDirectory() as temp_dir:
        rosettafold_json = Rosettafoldjson(temp_dir)

        rosettafold_json.json_to_json(test_data.test_inputAB_json)
        out_file = Path(temp_dir) / "rosettafold_output.json"
        rosettafold_json.write_json(out_file)

        reference = {
            'name': '2PV7',
            'components':
                [
                    {'seq': 'GMRES', 'chain_ids': 'A'},
                    {'seq': 'GMRES', 'chain_ids': 'B'},
                    {'seq': 'YANEN', 'chain_ids': 'C'},
                    {'ccd_code': 'ATP'},
                    {'ccd_code': 'ATP'},
                    {'smiles': 'CC(=O)OC1C[NH+]2CCC1CC2'}
                ]
        }

        with open(out_file, "r") as f:
            written_data = f.read()
        written_data = json.loads(written_data)


        assert written_data == reference
