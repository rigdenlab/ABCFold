import json
import tempfile
from pathlib import Path

from abcfold.protenix.af3_to_protenix import ProtenixJson

# flake8: noqa


def test_af3_to_protenix(test_data):
    with tempfile.TemporaryDirectory() as temp_dir:
        protenix_json = ProtenixJson(temp_dir)

        data = protenix_json.json_to_json(test_data.test_inputAB_json)

        reference = {
            'name': '2PV7',
            'modelSeeds': [1],
            'sequences':
                [
                    {'proteinChain': {'sequence': 'GMRES', 'count': 2}},
                    {'proteinChain': {'sequence': 'YANEN', 'count': 1}},
                    {'ligand': {'ligand': 'CCD_ATP', 'count': 2}},
                    {'ligand': {'ligand': 'CC(=O)OC1C[NH+]2CCC1CC2', 'count': 1}}
                ]
        }

        assert data == reference


def test_af3_to_protenix_rna(test_data):
    with tempfile.TemporaryDirectory() as temp_dir:
        protenix_json = ProtenixJson(temp_dir)

        data = protenix_json.json_to_json(test_data.test_inputRNA_json)

        reference = {
            'name': 'RNA_example',
            'modelSeeds': [1],
            'sequences':
                [
                    {'rnaSequence': {'sequence': 'AGCU', 'count': 1}}
                ]
        }

        assert data == reference


def test_af3_to_protenix_dna(test_data):
    with tempfile.TemporaryDirectory() as temp_dir:
        protenix_json = ProtenixJson(temp_dir)

        data = protenix_json.json_to_json(test_data.test_inputDNA_json)

        reference = {
            'name': 'DNA_example',
            'modelSeeds': [1],
            'sequences':
                [
                    {'dnaSequence': {'sequence': 'AGCT', 'count': 2}}
                ]
        }

        assert data == reference


def test_af3_to_protenix_ligand(test_data):
    with tempfile.TemporaryDirectory() as temp_dir:
        protenix_json = ProtenixJson(temp_dir)

        data = protenix_json.json_to_json(test_data.test_inputLIG_json)

        reference = {
            'name': '2PV7',
            'modelSeeds': [1],
            'sequences':
                [
                    {'proteinChain': {'sequence': 'GMRESYANENQFGFKTINSDIHKIVIVGGYGKLGGLFARYLRASGYPISILDREDWAVAESILANADVVIVSVPINLTLETIERLKPYLTENMLLADLTSVKREPLAKMLEVHTGAVLGLHPMFGADIASMAKQVVVRCDGRFPERYEWLLEQIQIWGAKIYQTNATEHDHNMTYIQALRHFSTFANGLHLSKQPINLANLLALSSPIYRLELAMIGRLFAQDAELYADIIMDKSENLAVIETLKQTYDEALTFFENNDRQGFIDAFHKVRDWFGDYSEQFLKESRQLLQQANDLKQG', 'count': 2}},
                    {'ligand': {'ligand': 'CCD_ATP', 'count': 2}},
                    {'ligand': {'ligand': 'CC(=O)OC1C[NH+]2CCC1CC2', 'count': 1}},
                    {'ligand': {'ligand': 'CCCCCCCCCCCC(O)=O', 'count': 2}},
                    {'ion': {'ion': 'CCD_MG', 'count': 1}}
                ]
            }

        assert data == reference


def test_af3_to_protenix_ptm(test_data):
    with tempfile.TemporaryDirectory() as temp_dir:
        protenix_json = ProtenixJson(temp_dir)

        data = protenix_json.json_to_json(test_data.test_inputPTM_json)

        reference = {
            'name': 'PTM example',
            'modelSeeds': [1],
            'sequences':
                [
                    {'proteinChain': {'sequence': 'PVLSCGEWQL',
                                      'count': 1,
                                      'modifications': [
                                          {'ptmType': 'CCD_HY3', 'ptmPosition': 1},
                                          {'ptmType': 'CCD_P1L', 'ptmPosition': 5}
                                      ]
                                     }
                    },
                    {'rnaSequence': {'sequence': 'AGCU',
                                     'count': 1,
                                     'modifications': [
                                         {'modificationType': 'CCD_2MG',
                                          'basePosition': 1},
                                         {'modificationType': 'CCD_5MC',
                                          'basePosition': 4}
                                     ]
                                    }
                    }
                ]
        }

        assert data == reference

def test_af3_to_protenix_constraints(test_data):
    with tempfile.TemporaryDirectory() as temp_dir:
        protenix_json = ProtenixJson(temp_dir)

        data = protenix_json.json_to_json(test_data.test_inputBOND_json)

        reference = {
            'name': '7SYZ',
            'modelSeeds': [1],
            'sequences': [
                {'proteinChain': {'sequence': 'MMADSKLVSLNNNLSGKIKDQGKVIKNYYGTMDIKKINDGLLDSKILGAFNTVIALLGSIIIIVMNIMIIQNYTRTTDNQALIKESLQSVQQQIKALTDKIGTEIGPKVSLIDTSSTITIPANIGLLGSKISQSTSSINENVNDKCKFTLPPLKIHECNISCPNPLPFREYRPISQGVSDLVGLPNQICLQKTTSTILKPRLISYTLPINTREGVCITDPLLAVDNGFFAYSHLEKIGSCTRGIAKQRIIGVGEVLDRGDKVPSMFMTNVWTPPNPSTIHHCSSTYHEDFYYTLCAVSHVGDPILNSTSWTESLSLIRLAVRPKSDSGDYNQKYIAITKVERGKYDKVMPYGPSGIKQGDTLYFPAVGFLPRTEFQYNDSNCPIIHCKYSKAENCRLSMGVNSKSHYILRSGLLKYNLSLGGDIILQFIEIADNRLTIGSPSKIYNSLGQPVFYQASYSWDTMIKLGDVDTVDPLRVQWRNNSVISRPGQSQCPRFNVCPEVCWEGTYNDAFLIDRLNWVSAGVYLNSNQTAENPVFAVFKDNEILYQVPLAEDDTNAQKTITDCFLLENVIWCISLVEIYDTGDSVIRPKLFAVKIPAQCSES', 'count': 1}},
                {'proteinChain': {'sequence': 'QIQLVQSGPELKKPGETVKISCTTSGYTFTNYGLNWVKQAPGKGFKWMAWINTYTGEPTYADDFKGRFAFSLETSASTTYLQINNLKNEDMSTYFCARSGYYDGLKAMDYWGQGTSVTVSSAKTTPPSVYPLAPGSAAQTNSMVTLGCLVKGYFPEPVTVTWNSGSLSSGVHTFPAVLQSDLYTLSSSVTVPSSTWPSETVTCNVAHPASSTKVDKKIVPRDC', 'count': 1}},
                {'proteinChain': {'sequence': 'DVLMIQTPLSLPVSLGDQASISCRSSQSLIHINGNTYLEWYLQKPGQSPKLLIYKVSNRFSGVPDRFSGSGSGTDFTLKISRVEAEDLGVYYCFQGSHVPFTFGAGTKLELKRADAAPTVSIFPPSSEQLTSGGASVVCFLNNFYPKDINVKWKIDGSERQNGVLNSWTDQDSKDSTYSMSSTLTLTKDEYERHNSYTCEATHKTSTSPIVKSFNRNECVY', 'count': 1}}
            ],
            "contact": [
                {
                    "entity1": 1,
                    "copy1": 1,
                    "position1": 387,
                    "atom1": "CA",
                    "entity2": 2,
                    "copy2": 1,
                    "position2": 101,
                    "atom2": "CA",
                    "max_distance": 6,
                    "min_distance": 0
                },
                {
                    "entity1": 3,
                    "copy1": 1,
                    "position1": 32,
                    "atom1": "CA",
                    "entity2": 1,
                    "copy2": 1,
                    "atom2": "CA",
                    "position2": 483,
                    "max_distance": 6,
                    "min_distance": 0
                }
            ]
        }

        assert data == reference

def test_protenix_output_msa(test_data):
    with tempfile.TemporaryDirectory() as temp_dir:
        protenix_json = ProtenixJson(temp_dir)

        data = protenix_json.json_to_json(test_data.test_inputAmsa_json)
        msa_path = (
            data["sequences"][0]["proteinChain"]
            .get("msa", {})
            .get("precomputed_msa_dir")
        )
        # MSA directory has a random path, so just check that it exists then give
        # it a placeholder value for comparison
        assert msa_path is not None
        assert Path(msa_path).exists()
        data["sequences"][0]["proteinChain"]["msa"]["precomputed_msa_dir"] = (
            "PRECOMPUTED_MSA_DIR"
        )

        reference = {
            'name': '2PV7',
            'modelSeeds': [1],
            'contact': [
                {'entity1': 1,
                 'position1': 1,
                 'copy1': 1, 'atom1':
                 'CA', 'entity2': 1,
                 'position2': 20,
                 'copy2': 1,
                 'atom2': 'CA',
                 'max_distance': 6,
                 'min_distance': 0}
            ],
            'sequences': [
                {'proteinChain': {'sequence': 'GMRESYANENQFGFKTINSDIHKIVIVGGYGKLGGLFARYLRASGYPISILDREDWAVAESILANADVVIVSVPINLTLETIERLKPYLTENMLLADLTSVKREPLAKMLEVHTGAVLGLHPMFGADIASMAKQVVVRCDGRFPERYEWLLEQIQIWGAKIYQTNATEHDHNMTYIQALRHFSTFANGLHLSKQPINLANLLALSSPIYRLELAMIGRLFAQDAELYADIIMDKSENLAVIETLKQTYDEALTFFENNDRQGFIDAFHKVRDWFGDYSEQFLKESRQLLQQANDLKQG',
                                  'count': 1,
                                  'msa': {'precomputed_msa_dir': 'PRECOMPUTED_MSA_DIR',
                                          'pairing_db': 'uniref100'
                                          }
                                }
                }
            ]
        }

        assert data == reference

def test_protenix_write_json(test_data):
    with tempfile.TemporaryDirectory() as temp_dir:
        protenix_json = ProtenixJson(temp_dir)

        protenix_json.json_to_json(test_data.test_inputAB_json)

        out_file = Path(temp_dir) / "protenix_output.json"
        protenix_json.write_json(out_file)

        reference = [
            {
            'name': '2PV7',
            'modelSeeds': [1],
            'sequences':
                [
                    {'proteinChain': {'sequence': 'GMRES', 'count': 2}},
                    {'proteinChain': {'sequence': 'YANEN', 'count': 1}},
                    {'ligand': {'ligand': 'CCD_ATP', 'count': 2}},
                    {'ligand': {'ligand': 'CC(=O)OC1C[NH+]2CCC1CC2', 'count': 1}}
                ]
            }
        ]

        with open(out_file, "r") as f:
            written_data = f.read()
        written_data = json.loads(written_data)


        assert written_data == reference
