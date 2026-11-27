import json
import logging
import random
import re
import string
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

logger = logging.getLogger("logger")


ION_CCD_CODES = [
    # alkali / alkaline earth
    "LI", "NA", "K", "RB", "CS",
    "MG", "CA", "SR", "BA",
    # transition metals
    "ZN", "FE", "FE2", "FE3", "CU", "MN", "CO", "NI",
    "CD", "HG",
    # halides
    "F", "CL", "BR", "I",
]


class ProtenixJson:
    """
    Object to convert an AlphaFold3 json file to a Protenix JSON file.
    """

    def __init__(self, working_dir: Union[str, Path], create_files: bool = True):
        self.working_dir = working_dir
        self.seeds: list = [42]
        self.__ids: Dict = {}
        self.__id_counter: int = 1
        self.__create_files = create_files
        self.protenix_dict: Dict = {}
        self.template_mmcif_dir: Optional[Path] = None
        self.release_dates_path: Optional[Path] = None
        self.obsolete_pdbs_path: Optional[Path] = None
        self._template_counter: int = 0
        self._release_dates: Dict[str, Dict[str, str]] = {}

    @staticmethod
    def _normalise_template_cif(
        mmcif: str, out_file: Path, default_date: str = "2000-01-01"
    ) -> Optional[Tuple[Optional[str], str, str, str]]:
        """Write a normalised mmCIF for Protenix's TemplateHitFeaturizer.

        Uses gemmi to regenerate ``_entity``/``_entity_poly_seq``/``_struct_asym``
        records (which BioPython-written mmCIFs omit) and preserves the real
        PDB ID and release date when present (falling back to ``None`` /
        ``default_date`` for custom templates).

        Returns:
            Tuple of ``(real_pdb_id_or_None, chain_id, one_letter_sequence,
            release_date)`` for the first polymer chain, or ``None`` if the
            mmCIF could not be parsed.
        """
        try:
            import gemmi

            doc = gemmi.cif.read_string(mmcif)
            in_block = doc[0]

            # Extract real 4-char PDB ID if present (_entry.id or data_ block)
            real_pdb_id: Optional[str] = None
            entry_val = in_block.find_value("_entry.id")
            for candidate in (
                entry_val.strip("'\" ") if entry_val else None,
                in_block.name.strip() if in_block.name else None,
            ):
                if candidate and re.fullmatch(r"[0-9][a-zA-Z0-9]{3}", candidate):
                    real_pdb_id = candidate.lower()
                    break

            # Extract earliest release date (YYYY-MM-DD) from input mmCIF
            rev_dates = [
                v.strip("'\" ")
                for v in in_block.find_values(
                    "_pdbx_audit_revision_history.revision_date"
                )
                if re.fullmatch(r"\d{4}-\d{2}-\d{2}", v.strip("'\" "))
            ]
            release_date = (
                min(rev_dates) if (real_pdb_id and rev_dates) else default_date
            )

            st = gemmi.make_structure_from_block(in_block)
            if len(st) == 0:
                raise ValueError("template mmCIF has no models")
            st.setup_entities()

            model0 = st[0]
            chain_id = None
            res_names: List[str] = []
            for entity in st.entities:
                if entity.entity_type.name != "Polymer":
                    continue
                if not entity.subchains:
                    continue
                first_subchain = entity.subchains[0]
                seq = [
                    residue.name
                    for chain in model0
                    for residue in chain
                    if residue.subchain == first_subchain
                ]
                if seq and not entity.full_sequence:
                    entity.full_sequence = seq
                if chain_id is None and seq:
                    for chain in model0:
                        if any(r.subchain == first_subchain for r in chain):
                            chain_id = chain.name
                            res_names = seq
                            break

            if not chain_id or not res_names:
                raise ValueError("template mmCIF has no polymer chain")

            out_doc = st.make_mmcif_document()
            out_block = out_doc[0]
            if not out_block.find_values(
                "_pdbx_audit_revision_history.revision_date"
            ):
                out_block.set_pair("_pdbx_audit_revision_history.ordinal", "1")
                out_block.set_pair(
                    "_pdbx_audit_revision_history.revision_date", release_date
                )
            out_doc.write_file(str(out_file))
            tmpl_seq = gemmi.one_letter_code(res_names)
            return real_pdb_id, chain_id, tmpl_seq, release_date
        except Exception as exc:  # noqa: BLE001
            logger.warning("Could not prepare Protenix template CIF: %s", exc)
            return None

    @staticmethod
    def _build_a3m_alignment(
        query_seq: str,
        tmpl_seq: str,
        query_indices: Optional[List[int]] = None,
        template_indices: Optional[List[int]] = None,
    ) -> Optional[Tuple[str, int, int]]:
        """Build an hmmsearch-style A3M alignment string from AF3 index maps.

        Returns:
            Tuple of ``(aligned_seq, start_1based, end_1based)`` or ``None`` if
            no valid aligned residues exist.
        """
        q_len = len(query_seq)
        t_len = len(tmpl_seq)
        if q_len == 0 or t_len == 0:
            return None

        if query_indices and template_indices:
            raw_pairs = sorted(
                (int(q), int(t))
                for q, t in zip(query_indices, template_indices)
                if 0 <= int(q) < q_len and 0 <= int(t) < t_len
            )
            # Keep strictly increasing (q, t) pairs for a valid linear A3M
            pairs: List[Tuple[int, int]] = []
            last_q, last_t = -1, -1
            for q, t in raw_pairs:
                if q > last_q and t > last_t:
                    pairs.append((q, t))
                    last_q, last_t = q, t
        else:
            n = min(q_len, t_len)
            pairs = [(i, i) for i in range(n)]

        if not pairs:
            return None

        q_to_t = dict(pairs)
        start_t = pairs[0][1]
        end_t = pairs[-1][1]
        expected_t = start_t
        chars: List[str] = []

        for q in range(q_len):
            if q not in q_to_t:
                chars.append("-")
            else:
                t = q_to_t[q]
                if t > expected_t:
                    chars.append(tmpl_seq[expected_t:t].lower())
                chars.append(tmpl_seq[t].upper())
                expected_t = t + 1

        return "".join(chars), start_t + 1, end_t + 1

    def _write_templates_a3m(
        self, query_seq: str, templates: list
    ) -> Optional[Path]:
        """Stage inline AF3 templates as local mmCIFs + a Protenix A3M file."""
        if not self.__create_files:
            return None

        if self.template_mmcif_dir is None:
            self.template_mmcif_dir = Path(self.working_dir) / "template_mmcif"
            self.template_mmcif_dir.mkdir(parents=True, exist_ok=True)

        a3m_entries: List[str] = []
        for template in templates:
            if not isinstance(template, dict):
                continue
            mmcif = template.get("mmcif")
            if not mmcif:
                continue

            # Write to a temporary path first, then rename to the real 4-char
            # PDB ID if present (or a synthetic xNNN ID for custom templates).
            tmp_cif = self.template_mmcif_dir / f"_tmp_{self._template_counter}.cif"
            parsed = self._normalise_template_cif(mmcif, tmp_cif)
            if parsed is None:
                tmp_cif.unlink(missing_ok=True)
                continue
            real_pdb_id, chain_id, tmpl_seq, release_date = parsed

            if real_pdb_id and real_pdb_id not in self._release_dates:
                pdb_id = real_pdb_id
            else:
                pdb_id = f"x{self._template_counter:03d}"
                self._template_counter += 1
            cif_file = self.template_mmcif_dir / f"{pdb_id}.cif"
            tmp_cif.replace(cif_file)

            aligned = self._build_a3m_alignment(
                query_seq,
                tmpl_seq,
                template.get("queryIndices"),
                template.get("templateIndices"),
            )
            if aligned is None:
                cif_file.unlink(missing_ok=True)
                continue
            aligned_seq, start, end = aligned

            self._release_dates[pdb_id] = {"release_date": release_date}
            header = (
                f">{pdb_id}_{chain_id}/{start}-{end} "
                f"mol:protein length:{len(tmpl_seq)}"
            )
            a3m_entries.append(f"{header}\n{aligned_seq}\n")

        if not a3m_entries:
            return None

        rand_tag = "".join(random.choices(string.ascii_letters, k=5))
        a3m_path = Path(self.working_dir) / f"templates_{rand_tag}.a3m"
        a3m_path.write_text("".join(a3m_entries))
        return a3m_path

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

    def json_to_json(
        self,
        json_file_or_dict: Union[dict, str, Path],
    ):
        """
        Main function to convert an AF3 json file or dict to a Protenix json string

        Args:
            json_file_or_dict (Union[dict, str, Path]): json file or dict

        Returns:
            Dict: protenix dictionary
        """
        logger.info("Converting input json to a Protenix compatible json file")
        if isinstance(json_file_or_dict, str) or isinstance(json_file_or_dict, Path):
            with open(json_file_or_dict, "r") as f:
                json_dict = json.load(f)
        else:
            json_dict = json_file_or_dict

        protenix_sequences = []
        for key, value in json_dict.items():
            if key == "name":
                self.protenix_dict["name"] = value
            if key == "modelSeeds":
                if isinstance(value, list):
                    self.seeds = value
                elif isinstance(value, int):
                    self.seeds = [value]
                self.protenix_dict["modelSeeds"] = self.seeds

            if key == "sequences":
                for entry in value:
                    if "protein" in entry:
                        protein_entry = self.convert_protein(entry["protein"])
                        protenix_sequences.append(protein_entry)
                    elif "rna" in entry:
                        rna_entry = self.convert_rna(entry["rna"])
                        protenix_sequences.append(rna_entry)
                    elif "dna" in entry:
                        dna_entry = self.convert_dna(entry["dna"])
                        protenix_sequences.append(dna_entry)
                    elif "ligand" in entry:
                        ligand_entry = self.convert_ligand(entry["ligand"])
                        protenix_sequences.append(ligand_entry)

            if key == "bondedAtomPairs" and value is not None:
                contact = self.convert_bonded_atom_pairs(value)
                self.protenix_dict["contact"] = contact

        self.protenix_dict["sequences"] = protenix_sequences

        if self._release_dates and self.__create_files:
            self.release_dates_path = (
                Path(self.working_dir) / "template_release_dates.json"
            )
            self.release_dates_path.write_text(json.dumps(self._release_dates))
            self.obsolete_pdbs_path = (
                Path(self.working_dir) / "template_obsolete_pdbs.json"
            )
            self.obsolete_pdbs_path.write_text("{}")

        return self.protenix_dict

    def convert_protein(self, seq_dict) -> Dict[str, Any]:
        sequence = seq_dict["sequence"]
        chain_ids = seq_dict.get("id", [])
        for chain in chain_ids:
            self.__ids[chain] = self.__id_counter
            self.__id_counter += 1
        count = len(chain_ids) if chain_ids else 1

        protein_chain = {
            "sequence": sequence,
            "count": count,
        }

        modifications = seq_dict.get("modifications")
        if modifications:
            prefixed = []
            for mod in modifications:
                m = dict(mod)
                typ = str(m.get("ptmType", ""))
                if not typ.startswith("CCD_"):
                    typ = "CCD_" + typ
                m["ptmType"] = typ
                prefixed.append(m)
            protein_chain["modifications"] = prefixed

        unpaired_msa = seq_dict.get("unpairedMsa")
        paired_msa = seq_dict.get("pairedMsa")
        random_string = ''.join(random.choices(string.ascii_letters, k=5))
        msa_dir = Path(self.working_dir) / random_string
        if unpaired_msa and self.__create_files:
            if not msa_dir.exists():
                msa_dir.mkdir(parents=True, exist_ok=True)
            self.msa_to_file(
                unpaired_msa,
                msa_dir / "non_pairing.a3m"
            )

        if paired_msa and self.__create_files:
            if not msa_dir.exists():
                msa_dir.mkdir(parents=True, exist_ok=True)
            self.msa_to_file(
                paired_msa,
                msa_dir / "pairing.a3m"
            )

        if unpaired_msa or paired_msa and self.__create_files:
            protein_chain["msa"] = {
                "precomputed_msa_dir": msa_dir.as_posix(),
                "pairing_db": "uniref100"
            }

        templates = seq_dict.get("templates")
        if templates and self.__create_files:
            a3m_path = self._write_templates_a3m(sequence, templates)
            if a3m_path is not None:
                protein_chain["templatesPath"] = a3m_path.resolve().as_posix()

        return {
            "proteinChain": protein_chain
        }

    def convert_rna(self, seq_dict) -> Dict[str, Any]:
        sequence = seq_dict["sequence"]
        chain_ids = seq_dict.get("id", [])
        for chain in chain_ids:
            self.__ids[chain] = self.__id_counter
            self.__id_counter += 1
        count = len(chain_ids) if chain_ids else 1

        rna_chain = {
            "sequence": sequence,
            "count": count,
        }

        modifications = seq_dict.get("modifications")
        if modifications:
            prefixed = []
            for mod in modifications:
                m = dict(mod)
                typ = str(m.get("modificationType", ""))
                if not typ.startswith("CCD_"):
                    typ = "CCD_" + typ
                m["modificationType"] = typ
                prefixed.append(m)
            rna_chain["modifications"] = prefixed

        return {
            "rnaSequence": rna_chain
        }

    def convert_dna(self, seq_dict) -> Dict[str, Any]:
        sequence = seq_dict["sequence"]
        chain_ids = seq_dict.get("id", [])
        for chain in chain_ids:
            self.__ids[chain] = self.__id_counter
            self.__id_counter += 1
        count = len(chain_ids) if chain_ids else 1

        dna_chain = {
            "sequence": sequence,
            "count": count,
        }

        modifications = seq_dict.get("modifications")
        if modifications:
            prefixed = []
            for mod in modifications:
                m = dict(mod)
                typ = str(m.get("modificationType", ""))
                if not typ.startswith("CCD_"):
                    typ = "CCD_" + typ
                m["modificationType"] = typ
                prefixed.append(m)
            dna_chain["modifications"] = prefixed

        return {
            "dnaSequence": dna_chain
        }

    def convert_ligand(self, seq_dict) -> Dict[str, Any]:
        chain_ids = seq_dict.get("id", [])
        for chain in chain_ids:
            self.__ids[chain] = self.__id_counter
            self.__id_counter += 1
        count = len(chain_ids) if chain_ids else 1

        if "ccdCodes" in seq_dict:
            ligand_id = seq_dict["ccdCodes"][0]
            ligand = f"CCD_{ligand_id}"
        else:
            ligand = seq_dict["smiles"]
            ligand_id = ligand

        if ligand_id.upper() in ION_CCD_CODES:
            ligand_chain = {
                "ion": ligand,
                "count": count,
            }
            return {
                "ion": ligand_chain
            }
        else:
            ligand_chain = {
                "ligand": ligand,
                "count": count,
            }
            return {
                "ligand": ligand_chain
            }

    def convert_bonded_atom_pairs(self,
                                  bonded_atom_pairs: List[List[List[Union[str, int]]]]):
        contacts = []
        for pair in bonded_atom_pairs:
            entity1, position1, atom1 = pair[0]
            entity2, position2, atom2 = pair[1]
            contact = {
                "entity1": self.__ids[entity1],
                "position1": position1,
                "copy1": 1,
                "atom1": atom1,
                "entity2": self.__ids[entity2],
                "position2": position2,
                "copy2": 1,
                "atom2": atom2,
                "max_distance": 6,
                "min_distance": 0
            }
            contacts.append(contact)
        return contacts

    def write_json(self, out_file: Union[str, Path]):
        """
        Write the Protenix json to a file

        Args:
            out_file (Union[str, Path]): output file path

        Returns:
            None
        """

        with open(out_file, "w") as f:
            json.dump([self.protenix_dict], f, indent=4)
