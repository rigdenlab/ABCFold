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


def test_af3_to_protenix_templates(test_data):
    import re

    minimal_cif = Path(test_data.test_6BJ9A_cif).read_text()
    # Extract a short sequence from the CIF using gemmi so query/template match
    import gemmi

    doc = gemmi.cif.read_string(minimal_cif)
    st = gemmi.make_structure_from_block(doc[0])
    st.setup_entities()
    chain0 = st[0][0]
    res_names = [r.name for r in chain0 if r.entity_type.name == "Polymer"]
    tmpl_seq = gemmi.one_letter_code(res_names)

    # Query has a gap at pos 0 and a skipped residue at template pos 3
    query_seq = "A" + tmpl_seq[:3] + tmpl_seq[4:10]
    q_idx = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    t_idx = [0, 1, 2, 4, 5, 6, 7, 8, 9]

    af3_dict = {
        "name": "tmpl_test",
        "modelSeeds": [42],
        "sequences": [
            {
                "protein": {
                    "id": ["A"],
                    "sequence": query_seq,
                    "templates": [
                        {
                            # Real PDB mmCIF (6BJ9) should preserve its PDB code
                            "mmcif": minimal_cif,
                            "queryIndices": q_idx,
                            "templateIndices": t_idx,
                        },
                        {
                            # Custom mmCIF without a 4-char PDB id gets x000
                            "mmcif": minimal_cif.replace("data_6BJ9", "data_custom")
                            .replace("6BJ9", "CUSTOM"),
                            "queryIndices": q_idx,
                            "templateIndices": t_idx,
                        },
                    ],
                }
            }
        ],
    }

    with tempfile.TemporaryDirectory() as temp_dir:
        pj = ProtenixJson(temp_dir)
        data = pj.json_to_json(af3_dict)

        pchain = data["sequences"][0]["proteinChain"]
        assert "templatesPath" in pchain
        a3m_path = Path(pchain["templatesPath"])
        assert a3m_path.exists()
        assert pj.template_mmcif_dir is not None
        assert (pj.template_mmcif_dir / "6bj9.cif").exists()
        assert (pj.template_mmcif_dir / "x000.cif").exists()
        assert pj.release_dates_path is not None
        rel_dates = json.loads(pj.release_dates_path.read_text())
        assert "6bj9" in rel_dates
        assert rel_dates["x000"]["release_date"] == "2000-01-01"
        assert pj.obsolete_pdbs_path is not None
        assert json.loads(pj.obsolete_pdbs_path.read_text()) == {}

        # Verify the A3M matches Protenix's HmmsearchA3MParser and
        # get_pdb_id_and_chain regexes and indexing
        lines = a3m_path.read_text().strip().splitlines()
        assert len(lines) == 4
        header, aligned, custom_header, _ = lines
        assert "mol:protein" in header
        pattern = r"^>?([a-z0-9]+)_(\w+)/([0-9]+)-([0-9]+).*protein length:([0-9]+) *(.*)$"
        m = re.match(pattern, header)
        assert m is not None
        assert m[1] == "6bj9"
        assert re.match(r"([a-zA-Z\d]{4})_([a-zA-Z0-9.]+)", f"{m[1]}_{m[2]}") is not None
        assert int(m[3]) == 1
        assert int(m[4]) == 10
        assert int(m[5]) == len(tmpl_seq)

        m_custom = re.match(pattern, custom_header)
        assert m_custom is not None
        assert m_custom[1] == "x000"

        # Trace HmmsearchA3MParser._get_indices(aligned, start - 1)
        indices = []
        counter = int(m[3]) - 1
        for ch in aligned:
            if ch == "-":
                indices.append(-1)
            elif ch.islower():
                counter += 1
            else:
                indices.append(counter)
                counter += 1
        assert len(indices) == len(query_seq)
        assert indices == [-1] + t_idx
