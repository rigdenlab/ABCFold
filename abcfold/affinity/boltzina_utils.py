# This module collects the Boltzina-specific helper functions used by the
# standalone ``abcfold/scripts/boltz_affinity.py`` entry point.
#
# Background
# ----------
# Boltzina (https://github.com/ohuelab/boltzina) scores a protein-ligand
# complex with the Boltz-2 affinity head.  Upstream Boltzina *places* the
# ligand itself with AutoDock Vina and then reuses a Boltz "processed"
# work_dir (``processed/manifest.json`` + ``processed/constraints/*.npz``) to
# run only the affinity module of Boltz.
#
# In ABCFold the ligand is already placed for us by the structure-prediction
# back-ends (AlphaFold3, Chai-1, Protenix, Boltz, ...).  Each of those writes a
# *complex* mmCIF that contains the protein together with the docked ligand.
# We therefore drop the Vina docking stage entirely and feed the predicted
# complex straight into the Boltz affinity scorer -- conceptually the same as
# Boltzina's ``scoring_only`` path.
#
# The functions below cover everything that path needs:
#   * structural auto-detection of the non-polymer ligand,
#   * extraction of the protein sequence(s),
#   * building an RDKit mol (with atom names) for the ligand,
#   * building (or locating) the Boltz ``processed`` work_dir,
#   * turning a complex mmCIF into a ``pre_affinity_*.npz`` structure file,
#   * wiring up the per-record manifest / constraints,
#   * collecting the affinity JSON results.
#
# Heavy Boltz / Boltzina / RDKit imports are performed lazily inside the
# functions that need them, so this module can be imported (and syntax-checked)
# outside the Boltz micromamba environment.

import copy
import json
import logging
import os
import pickle
import shutil
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger("logger")

# Residue names we never treat as a scorable ligand.
WATER_RESNAMES = {"HOH", "WAT", "DOD", "H2O"}
STANDARD_NUCLEOTIDES = {
    "A", "C", "G", "U", "T", "I",
    "DA", "DC", "DG", "DT", "DI", "DU",
    "RA", "RC", "RG", "RU",
}
# Common crystallisation / buffer additives that are usually not the ligand of
# interest.  Auto-detection still reports them, but they are de-prioritised.
COMMON_ADDITIVES = {
    "SO4", "PO4", "GOL", "EDO", "PEG", "ACT", "CL", "NA", "MG", "ZN",
    "CA", "K", "MN", "FE", "CO", "NI", "CU", "IOD", "BR", "FMT", "DMS",
}


# --------------------------------------------------------------------------- #
# Version-robust calling of Boltz internals
# --------------------------------------------------------------------------- #
def _call_with_supported(func, **candidates):
    """Call ``func`` passing only the candidate kwargs its signature accepts.

    Boltz's processing helpers have churned across releases (renamed/added
    arguments, positional-only params). We introspect the real signature and
    forward only the names it declares, so the same code works across versions
    without hard-coding a single signature.
    """
    import inspect

    try:
        sig = inspect.signature(func)
    except (TypeError, ValueError):
        # Builtins / C funcs with no signature: best-effort positional call.
        return func(candidates.get("data"))

    params = sig.parameters
    accepts_var_kw = any(
        p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()
    )

    kwargs = {}
    for name, value in candidates.items():
        if name in params or accepts_var_kw:
            kwargs[name] = value

    missing = [
        name
        for name, p in params.items()
        if p.default is inspect.Parameter.empty
        and p.kind
        in (
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
        )
        and name not in kwargs
    ]
    if missing:
        raise TypeError(
            f"{func.__module__}.{func.__name__} needs argument(s) {missing} "
            f"that boltzina_utils does not know how to supply. Inspect the "
            f"signature in your Boltz build: {sig}"
        )
    return func(**kwargs)


# --------------------------------------------------------------------------- #
# Boltz cache path (version-robust)
# --------------------------------------------------------------------------- #
def resolve_cache_path(cache_dir: Optional[Path] = None) -> Path:
    """Return the Boltz cache directory (holds ``ccd.pkl`` and ``mols/``).

    ``boltz.main.get_cache_path`` has moved/disappeared across Boltz versions,
    so we resolve defensively rather than relying on that symbol:

      1. an explicit ``cache_dir`` argument,
      2. ``boltz.main.get_cache_path`` if it happens to exist,
      3. the ``BOLTZ_CACHE`` environment variable (used by the Boltz CLI),
      4. the default ``~/.boltz``.
    """
    if cache_dir is not None:
        return Path(cache_dir).expanduser()
    try:
        from boltz.main import get_cache_path  # type: ignore

        return Path(get_cache_path())
    except Exception:  # noqa: BLE001
        pass
    env = os.environ.get("BOLTZ_CACHE")
    if env:
        return Path(env).expanduser()
    return Path("~/.boltz").expanduser()


def ensure_boltz_cache(cache_dir: Path) -> None:
    """Make sure the Boltz molecule/CCD cache is downloaded.

    The ``boltz predict`` CLI downloads ``ccd.pkl`` / ``mols/`` / weights into
    the cache on first use. We call ``process_inputs`` directly and so bypass
    that step -- which surfaces later as ``CCD component ALA not found!`` when
    ``load_canonicals`` cannot find the canonical residue mols. Here we invoke
    Boltz's own downloader (name varies by version) to populate the cache. It is
    idempotent: anything already present is skipped.
    """
    cache_dir = Path(cache_dir).expanduser()
    cache_dir.mkdir(parents=True, exist_ok=True)

    # If the canonical mols are already there, nothing to do.
    mol_dir = cache_dir / "mols"
    if mol_dir.exists() and any(mol_dir.iterdir()):
        return

    from boltz import main as bm

    for fn_name in ("download_boltz2", "download_boltz1", "download"):
        fn = getattr(bm, fn_name, None)
        if fn is None:
            continue
        try:
            logger.info("Populating Boltz cache via boltz.main.%s ...", fn_name)
            _call_with_supported(
                fn, cache=cache_dir, cache_dir=cache_dir, cache_path=cache_dir
            )
            return
        except Exception as exc:  # noqa: BLE001
            logger.warning("boltz.main.%s failed: %s", fn_name, exc)

    logger.warning(
        "Could not auto-populate the Boltz cache at %s. Run `boltz predict` "
        "once (which downloads ccd/mols), set BOLTZ_CACHE / --cache to an "
        "already-populated cache, or use --work_dir from an existing run.",
        cache_dir,
    )


# --------------------------------------------------------------------------- #
# Ligand auto-detection / structure inspection
# --------------------------------------------------------------------------- #
def _is_amino_acid(resname: str) -> bool:
    from Bio.PDB.Polypeptide import is_aa

    return is_aa(resname, standard=False)


def detect_ligands(
    model,
    include_additives: bool = False,
) -> List[Dict[str, Any]]:
    """Structurally auto-detect non-polymer ligand residues in a model.

    Unlike :meth:`CifFile.check_ligand`, this does *not* rely on the original
    input config (``input_params``) being present -- it inspects the structure
    directly, which is what we need for an arbitrary predicted complex.

    Args:
        model: A BioPython ``Model`` object (e.g. ``CifFile.model[0]``).
        include_additives: If ``True`` common buffer/ion additives are returned
            as candidate ligands as well.

    Returns:
        A list of dicts, one per detected ligand residue, each with keys
        ``chain_id``, ``resname``, ``resseq`` and ``num_atoms``.  Ordered with
        the most "ligand-like" candidate (largest non-additive heteromolecule)
        first.
    """
    ligands: List[Dict[str, Any]] = []
    for chain in model:
        for residue in chain:
            hetflag = residue.id[0]
            resname = residue.get_resname().strip().upper()

            # Standard polymer residue (blank hetero flag) -> not a ligand.
            if hetflag == " " or hetflag == "":
                continue
            if resname in WATER_RESNAMES:
                continue
            if resname in STANDARD_NUCLEOTIDES:
                continue
            if _is_amino_acid(resname):
                # Modified/HETATM amino acid that is still part of the polymer.
                continue
            if (not include_additives) and resname in COMMON_ADDITIVES:
                continue

            atoms = list(residue.get_atoms())
            num_heavy = sum(
                1
                for a in atoms
                if (getattr(a, "element", "") or "").strip().upper()
                not in ("H", "D")
            )
            ligands.append(
                {
                    "chain_id": chain.id,
                    "resname": resname,
                    "resseq": residue.id[1],
                    "num_atoms": len(atoms),
                    "num_heavy_atoms": num_heavy,
                }
            )

    # Biggest heteromolecule first -- a reasonable default "ligand of interest".
    ligands.sort(key=lambda x: x["num_heavy_atoms"], reverse=True)
    return ligands


def smiles_heavy_atom_count(smiles: str) -> Optional[int]:
    """Return the heavy-atom count of a SMILES string (None if unparseable)."""
    from rdkit import Chem

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return mol.GetNumAtoms()  # RDKit SMILES mols are heavy-atom only by default


def select_ligand(
    model,
    ligand_chain: Optional[str] = None,
    smiles: Optional[str] = None,
) -> Dict[str, Any]:
    """Resolve which ligand to score.

    ``ligand_chain`` restricts selection to one chain. When it isn't given and
    ``smiles`` is provided, candidates are disambiguated by matching the SMILES
    heavy-atom count -- this correctly picks the intended ligand over cofactors
    (e.g. ATP) that the size heuristic would otherwise grab. Otherwise the
    largest non-additive heteromolecule is returned.
    """
    candidates = detect_ligands(model, include_additives=True)
    if not candidates:
        raise ValueError(
            "No non-polymer ligand could be detected in the input structure. "
            "Pass --ligand_chain explicitly if the ligand is encoded unusually."
        )

    filtered = candidates
    if ligand_chain is not None:
        filtered = [c for c in filtered if c["chain_id"] == ligand_chain]

    if not filtered:
        raise ValueError(
            f"No ligand found in chain {ligand_chain!r}. Detected candidates: "
            f"{[(c['chain_id'], c['resname']) for c in candidates]}"
        )

    # SMILES-aware disambiguation when no explicit chain was given.
    if smiles and ligand_chain is None and len(filtered) > 1:
        target = smiles_heavy_atom_count(smiles)
        if target is not None:
            matches = [c for c in filtered if c["num_heavy_atoms"] == target]
            if matches:
                if len({c["resname"] for c in matches}) > 1:
                    logger.warning(
                        "Multiple ligands match the SMILES heavy-atom count "
                        "(%d): %s. Scoring %s/%s; pass --ligand_chain to choose.",
                        target,
                        [(c["chain_id"], c["resname"]) for c in matches],
                        matches[0]["chain_id"],
                        matches[0]["resname"],
                    )
                filtered = matches
            else:
                logger.warning(
                    "No detected ligand matches the SMILES heavy-atom count "
                    "(%d); candidates=%s. Falling back to the size heuristic -- "
                    "pass --ligand_chain if this is wrong.",
                    target,
                    [(c["chain_id"], c["resname"], c["num_heavy_atoms"])
                     for c in filtered],
                )

    # Prefer a non-additive candidate when auto-selecting.
    non_additive = [
        c for c in filtered if c["resname"] not in COMMON_ADDITIVES
    ]
    chosen = (non_additive or filtered)[0]
    if len(filtered) > 1:
        logger.warning(
            "Multiple ligand candidates found %s; scoring %s/%s. Use "
            "--ligand_chain to choose another.",
            [(c["chain_id"], c["resname"]) for c in filtered],
            chosen["chain_id"],
            chosen["resname"],
        )
    return chosen


def extract_protein_sequences(model) -> Dict[str, str]:
    """Return ``{chain_id: one_letter_sequence}`` for every polymer chain."""
    from Bio.SeqUtils import seq1

    sequences: Dict[str, str] = {}
    for chain in model:
        residues = [
            res
            for res in chain
            if res.id[0] == " " and _is_amino_acid(res.get_resname())
        ]
        if not residues:
            continue
        sequences[chain.id] = "".join(
            seq1(res.get_resname()) for res in residues
        )
    return sequences


# --------------------------------------------------------------------------- #
# Ligand -> RDKit mol
# --------------------------------------------------------------------------- #
def write_ligand_pdb(
    model,
    ligand: Dict[str, Any],
    out_pdb: Path,
    safe_resname: str = "LIG",
) -> Path:
    """Write the chosen ligand residue to a clean, strictly-columned PDB.

    Prediction back-ends label ligands inconsistently (long resnames like
    ``LIG0``, 4-character atom names, odd residue numbers), and BioPython's
    ``PDBIO`` faithfully reproduces those quirks -- which then make
    ``RDKit.MolFromPDBFile`` choke ("Problem with residue number ..."). Rather
    than fight that, we emit the heavy atoms ourselves with fixed-width HETATM
    records (safe resname ``LIG``, chain ``A``, resSeq ``1``, sequential
    serials, explicit element column), so RDKit can always parse it. Atom names
    -- which is what we need from the resulting mol -- are preserved.
    """
    chain_id = ligand["chain_id"]
    resseq = ligand["resseq"]
    resname = (safe_resname or "LIG")[:3].ljust(3)

    residue = None
    for chain in model:
        if chain.id != chain_id:
            continue
        for res in chain:
            if res.id[1] == resseq:
                residue = res
                break
        if residue is not None:
            break
    if residue is None:
        raise ValueError(
            f"Ligand residue {chain_id}/{resseq} not found in structure."
        )

    lines = []
    atom_names = []
    serial = 0
    for atom in residue.get_atoms():
        element = (getattr(atom, "element", "") or "").strip()
        if element.upper() in ("H", "D"):
            continue  # heavy atoms only (matches the SMILES template)
        name = atom.get_name().strip()
        if not element:
            element = "".join(c for c in name if c.isalpha())[:2] or "C"
        element = element.upper()
        serial += 1
        atom_names.append(name)
        # Atom-name column (13-16): 2-char elements left-justified from col 13,
        # 1-char elements indented one space (PDB convention).
        if len(element) == 2:
            atom_name = name[:4].ljust(4)
        else:
            atom_name = (" " + name)[:4].ljust(4)
        x, y, z = (float(c) for c in atom.coord)
        lines.append(
            "HETATM"
            + f"{serial:>5}"
            + " "
            + atom_name
            + " "
            + resname
            + " A"
            + f"{1:>4}"
            + "    "
            + f"{x:>8.3f}{y:>8.3f}{z:>8.3f}"
            + f"{1.0:>6.2f}{0.0:>6.2f}"
            + " " * 10
            + f"{element:>2}"
        )
    lines.append("END")

    out_pdb = Path(out_pdb)
    out_pdb.write_text("\n".join(lines) + "\n")
    return out_pdb, atom_names


def ligand_to_mol(
    model,
    ligand: Dict[str, Any],
    smiles: Optional[str] = None,
    work_dir: Optional[Path] = None,
):
    """Build an RDKit ``Mol`` for the ligand, preserving PDB atom names.

    Args:
        model: BioPython model containing the ligand.
        ligand: A ligand dict from :func:`detect_ligands` / :func:`select_ligand`.
        smiles: Optional SMILES used to assign correct bond orders to the
            coordinates (recommended -- PDB/mmCIF rarely encode bond orders).
        work_dir: Directory for the temporary ligand PDB. Defaults to a temp dir.

    Returns:
        An RDKit ``Mol`` with a single conformer and per-atom ``name`` props,
        matching how Boltzina prepares ligand mols.
    """
    from rdkit import Chem
    from rdkit.Chem import AllChem

    cleanup = False
    if work_dir is None:
        work_dir = Path(tempfile.mkdtemp(prefix="boltzina_lig_"))
        cleanup = True
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    lig_pdb = work_dir / f"ligand_{ligand['chain_id']}_{ligand['resseq']}.pdb"

    try:
        _, atom_names = write_ligand_pdb(model, ligand, lig_pdb)

        mol = Chem.MolFromPDBFile(str(lig_pdb), removeHs=False, sanitize=False)
        if mol is None:
            raise ValueError(
                f"RDKit failed to read ligand from {lig_pdb}. Provide --smiles "
                f"for ligand {ligand['resname']} to disambiguate."
            )

        if smiles:
            template = Chem.MolFromSmiles(smiles)
            if template is None:
                raise ValueError(f"Could not parse --smiles {smiles!r}")
            try:
                mol = AllChem.AssignBondOrdersFromTemplate(template, mol)
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "Failed to assign bond orders from SMILES (%s); continuing "
                    "with perceived bonds.",
                    exc,
                )
        try:
            Chem.SanitizeMol(mol)
        except Exception as exc:  # noqa: BLE001
            logger.warning("RDKit sanitisation warning for ligand: %s", exc)

        # Set the FULL original atom names by order (not the round-tripped PDB
        # names, which truncate to 4 chars -- e.g. Chai's "CL1_1" -> "CL1_").
        # parse_mmcif matches the ligand to the CIF by exact atom name, so the
        # names must equal the structure's verbatim.
        if mol.GetNumAtoms() == len(atom_names):
            for atom, name in zip(mol.GetAtoms(), atom_names):
                atom.SetProp("name", name)
        else:
            logger.warning(
                "Ligand atom count (%d) != structure heavy atoms (%d); falling "
                "back to PDB names.", mol.GetNumAtoms(), len(atom_names),
            )
            for atom in mol.GetAtoms():
                pdb_info = atom.GetPDBResidueInfo()
                if pdb_info is not None:
                    atom.SetProp("name", pdb_info.GetName().strip())
        return mol
    finally:
        if cleanup:
            shutil.rmtree(work_dir, ignore_errors=True)


def write_extra_mols(
    mol,
    record_ids: List[str],
    base_extra_mols_dir: Path,
    ligand_name: str = "LIG",
) -> None:
    """Pickle the ligand mol for each record id (Boltz ``extra_mols`` format).

    Boltz expects one ``{record_id}.pkl`` per record, each a
    ``{ligand_name: rdkit_mol}`` dict, located in the processed ``mols`` dir.
    """
    from rdkit import Chem

    Chem.SetDefaultPickleProperties(Chem.PropertyPickleOptions.AllProps)
    base_extra_mols_dir = Path(base_extra_mols_dir)
    base_extra_mols_dir.mkdir(parents=True, exist_ok=True)
    for record_id in record_ids:
        with open(base_extra_mols_dir / f"{record_id}.pkl", "wb") as fh:
            pickle.dump({ligand_name: mol}, fh)


def write_parse_mol(mol, ligand_name: str, parse_mols_dir: Path) -> Path:
    """Write the mol in the layout ``parse_mmcif`` expects.

    ``parse_mmcif(..., moldir=...)`` looks up a non-CCD residue by name, loading
    ``{moldir}/{resname}.pkl`` as a *bare* RDKit mol (note: a single mol, not a
    ``{name: mol}`` dict -- that dict layout is what the affinity *datamodule*
    consumes via :func:`write_extra_mols`). The two layouts are different and
    both are required.
    """
    from rdkit import Chem

    Chem.SetDefaultPickleProperties(Chem.PropertyPickleOptions.AllProps)
    parse_mols_dir = Path(parse_mols_dir)
    parse_mols_dir.mkdir(parents=True, exist_ok=True)
    out = parse_mols_dir / f"{ligand_name}.pkl"
    with open(out, "wb") as fh:
        pickle.dump(mol, fh)
    return out


def relabel_ligand_in_cif(
    cif_in: Path,
    old_resname: str,
    new_resname: str,
    cif_out: Path,
) -> Path:
    """Rename a ligand residue inside an mmCIF's ``_atom_site`` loop.

    ``parse_mmcif`` matches the CIF ligand to its mol by residue name, so the
    CIF ligand name must equal the canonical ligand name used for the mol
    pickles and the manifest. Prediction back-ends label ligands arbitrarily
    (e.g. ``LIG0``), so we rewrite just the component-id column(s) of the
    ``_atom_site`` loop for rows whose comp_id == ``old_resname``. All other
    content is preserved verbatim.
    """
    cif_in = Path(cif_in)
    cif_out = Path(cif_out)
    old = old_resname.strip()
    new = new_resname.strip()

    lines = cif_in.read_text().splitlines()
    out_lines: List[str] = []

    in_atom_site = False
    header: List[str] = []
    comp_id_cols: List[int] = []

    def _flush_header_state():
        return [c for c in header if c.endswith("comp_id")]

    i = 0
    n = len(lines)
    while i < n:
        line = lines[i]
        stripped = line.strip()

        if stripped == "loop_":
            # Look ahead: is the next header block an _atom_site loop?
            j = i + 1
            block: List[str] = []
            while j < n and lines[j].strip().startswith("_"):
                block.append(lines[j].strip())
                j += 1
            if block and block[0].startswith("_atom_site."):
                in_atom_site = True
                header = block
                comp_id_idx = [
                    idx
                    for idx, col in enumerate(header)
                    if col.split(".", 1)[-1].endswith("comp_id")
                ]
                comp_id_cols = comp_id_idx
                out_lines.append(line)
                out_lines.extend(lines[i + 1: j])
                i = j
                continue
            else:
                in_atom_site = False
                out_lines.append(line)
                i += 1
                continue

        if in_atom_site:
            # End of the data block?
            if stripped == "" or stripped == "#" or stripped.startswith("_") \
                    or stripped == "loop_":
                in_atom_site = False
                out_lines.append(line)
                i += 1
                continue
            tokens = stripped.split()
            if len(tokens) == len(header):
                changed = False
                for idx in comp_id_cols:
                    if tokens[idx] == old:
                        tokens[idx] = new
                        changed = True
                out_lines.append(" ".join(tokens) if changed else line)
            else:
                out_lines.append(line)
            i += 1
            continue

        out_lines.append(line)
        i += 1

    cif_out.write_text("\n".join(out_lines) + "\n")
    return cif_out


def normalize_complex_cif(
    cif_in: Path,
    cif_out: Path,
    ligand_chain: Optional[str] = None,
    old_resname: Optional[str] = None,
    new_resname: Optional[str] = None,
    remove_residues: Optional[List] = None,
) -> Path:
    """Normalise a predicted complex CIF for boltzina's ``parse_mmcif``.

    boltzina's parser reads gemmi ``structure.entities`` and indexes them by
    subchain id (``entities[subchain_id]``). Many prediction back-ends (e.g.
    OpenFold) emit CIFs without proper ``_entity``/``_struct_asym`` records, so
    those entities are missing and the parser raises ``KeyError`` on a chain id.
    Upstream Boltzina sidesteps this by routing structures through ``maxit``;
    we instead use gemmi's ``setup_entities()`` to regenerate the entity /
    subchain records from the atom records (idempotent when they already
    exist), and rewrite a clean mmCIF.

    ``remove_residues`` is a list of ``(chain_id, resseq)`` tuples for other
    ligands/cofactors to delete: ``parse_mmcif`` parses the *whole* complex and
    needs a mol for every non-CCD residue, so any non-selected ligand must be
    stripped. Removing them also makes the parsed structure match the manifest,
    which only describes the protein plus the single scored ligand. Waters are
    always removed.

    Optionally renames the selected ligand residue (restricted to
    ``ligand_chain`` when given) to ``new_resname`` in the same pass, so the CIF
    ligand name matches the mol pickles and manifest.
    """
    import gemmi

    st = gemmi.read_structure(str(cif_in))
    st.remove_waters()

    # Strip non-selected ligands/cofactors/ions.
    if remove_residues:
        remove_set = {(str(c), int(r)) for c, r in remove_residues}
        for model in st:
            for chain in model:
                doomed = [
                    i
                    for i, residue in enumerate(chain)
                    if (chain.name, residue.seqid.num) in remove_set
                ]
                for i in reversed(doomed):
                    del chain[i]
        st.remove_empty_chains()

    if old_resname and new_resname and old_resname.upper() != new_resname.upper():
        old_upper = old_resname.strip().upper()
        for model in st:
            for chain in model:
                if ligand_chain is not None and chain.name != ligand_chain:
                    continue
                for residue in chain:
                    # Case-insensitive: predictors may use lowercase resnames
                    # (e.g. Protenix "l01") while detection upper-cases them.
                    if residue.name.strip().upper() == old_upper:
                        residue.name = new_resname

    # Regenerate entity / subchain records so every subchain maps to an entity.
    st.setup_entities()

    # gemmi's setup_entities() assigns entity types + subchains but leaves
    # `full_sequence` empty. boltzina's parse_polymer aligns the modeled
    # residues against `entity.full_sequence`, so an empty sequence yields zero
    # matches (or an index overrun). Populate each polymer entity's sequence
    # from its first subchain's modeled residues.
    model0 = st[0]
    for entity in st.entities:
        if entity.entity_type.name != "Polymer" or entity.full_sequence:
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
        if seq:
            entity.full_sequence = seq

    cif_out = Path(cif_out)
    cif_out.parent.mkdir(parents=True, exist_ok=True)
    st.make_mmcif_document().write_file(str(cif_out))
    return cif_out


# --------------------------------------------------------------------------- #
# Boltz "processed" work_dir: build internally or reuse an existing one
# --------------------------------------------------------------------------- #
def build_boltz_yaml(
    sequences: Dict[str, str],
    ligand_name: str = "LIG",
    ligand_smiles: Optional[str] = None,
    ligand_ccd: Optional[str] = None,
    msa: Optional[Path] = None,
    out_yaml: Optional[Path] = None,
) -> Path:
    """Write a minimal Boltz-2 YAML for the complex, flagging affinity.

    The ligand is given its own chain and marked as the affinity binder. Either
    ``ligand_smiles`` or ``ligand_ccd`` must be supplied so Boltz can build the
    ligand topology when processing inputs. If ``msa`` (an .a3m path) is given,
    it is attached to every protein chain so Boltz reuses it instead of querying
    the MSA server.
    """
    if not ligand_smiles and not ligand_ccd:
        raise ValueError(
            "Provide ligand_smiles or ligand_ccd to build the Boltz YAML."
        )

    if out_yaml is None:
        out_yaml = Path(tempfile.mkstemp(suffix=".yaml")[1])
    out_yaml = Path(out_yaml)
    msa_path = str(Path(msa).resolve()) if msa else None

    lines: List[str] = ["version: 1", "sequences:"]
    used_ids = sorted(sequences.keys())
    for chain_id, seq in sequences.items():
        lines.append("  - protein:")
        lines.append(f"      id: {chain_id}")
        lines.append(f"      sequence: {seq}")
        if msa_path:
            lines.append(f"      msa: {msa_path}")

    # Give the ligand a fresh single-letter chain id not used by the protein.
    ligand_chain = next(
        (c for c in "BCDEFGHIJKLMNOPQRSTUVWXYZ" if c not in used_ids), "L"
    )
    lines.append("  - ligand:")
    lines.append(f"      id: {ligand_chain}")
    if ligand_smiles:
        lines.append(f"      smiles: '{ligand_smiles}'")
    else:
        lines.append(f"      ccd: {ligand_ccd}")

    lines.append("properties:")
    lines.append("  - affinity:")
    lines.append(f"      binder: {ligand_chain}")

    out_yaml.write_text("\n".join(lines) + "\n")
    return out_yaml


def build_processed_inputs(
    yaml_path: Path,
    work_dir: Path,
    cache_dir: Optional[Path] = None,
    use_msa_server: bool = True,
    msa_server_url: str = "https://api.colabfold.com",
    msa_pairing_strategy: str = "greedy",
    preprocessing_threads: int = 1,
) -> Path:
    """Build a Boltz ``processed`` work_dir from a YAML config.

    This is the "build it internally" path. It reuses Boltz-2's own input
    processing so the resulting ``manifest.json`` / ``constraints`` are exactly
    what the affinity module expects.

    NB: this requires the Boltz micromamba environment (boltz==2.2.1) and, for
    MSA generation, network access to the MSA server (unless an MSA is already
    available). This is the single integration point most likely to need
    tweaking against a specific Boltz build, so it is deliberately isolated.

    Args:
        yaml_path: Boltz YAML produced by :func:`build_boltz_yaml`.
        work_dir: Destination dir; ``work_dir/processed`` is created.
        cache_dir: Boltz cache (defaults to ``get_cache_path()``), holds CCD +
            mol cache used during processing.

    Returns:
        ``work_dir`` (with a populated ``processed`` subdirectory).
    """
    from boltz.main import check_inputs, process_inputs

    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    cache_dir = resolve_cache_path(cache_dir)
    ensure_boltz_cache(cache_dir)
    ccd_path = cache_dir / "ccd.pkl"
    mol_dir = cache_dir / "mols"

    # `check_inputs` / `process_inputs` signatures differ across Boltz versions
    # (arg names, positional `outdir`, presence of `mol_dir`/`boltz2`, ...).
    # Rather than hard-code one version, introspect each function and pass only
    # the arguments it actually accepts. `_call_with_supported` maps a pool of
    # candidate values (incl. common aliases) onto the real parameter names.
    data = _call_with_supported(
        check_inputs,
        data=Path(yaml_path),
        outdir=work_dir,
        out_dir=work_dir,
        override=False,
    )
    if data is None:
        # Some versions write in place and return None; fall back to the yaml.
        data = [Path(yaml_path)]

    _call_with_supported(
        process_inputs,
        data=data,
        out_dir=work_dir,
        outdir=work_dir,
        ccd_path=ccd_path,
        ccd=ccd_path,
        mol_dir=mol_dir,
        moldir=mol_dir,
        boltz2=True,
        use_msa_server=use_msa_server,
        msa_server_url=msa_server_url,
        msa_pairing_strategy=msa_pairing_strategy,
        preprocessing_threads=preprocessing_threads,
        max_msa_seqs=8192,
        override=False,
    )
    return work_dir


def load_manifest(work_dir: Path) -> dict:
    manifest_path = Path(work_dir) / "processed" / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"No Boltz manifest at {manifest_path}. Either pass a valid "
            "--work_dir from a prior Boltz run, or omit it to build one."
        )
    with open(manifest_path) as fh:
        return json.load(fh)


# --------------------------------------------------------------------------- #
# Per-record manifest / constraints / structure preparation
# --------------------------------------------------------------------------- #
def build_record_manifest(
    base_manifest: dict,
    base_record_id: str,
    record_ids: List[str],
    out_manifest: Path,
) -> Path:
    """Clone the base record into one record per ``record_ids`` entry."""
    manifest = copy.deepcopy(base_manifest)
    matches = [r for r in manifest["records"] if r["id"] == base_record_id]
    if not matches:
        # Fall back to the first record if ids don't line up.
        matches = manifest["records"][:1]
    template = matches[0]

    manifest["records"] = []
    for record_id in record_ids:
        new_record = copy.deepcopy(template)
        new_record["id"] = record_id
        manifest["records"].append(new_record)

    out_manifest = Path(out_manifest)
    out_manifest.parent.mkdir(parents=True, exist_ok=True)
    with open(out_manifest, "w") as fh:
        json.dump(manifest, fh, indent=4)
    return out_manifest


def link_constraints(
    source_work_dir: Path,
    base_record_id: str,
    record_ids: List[str],
    target_constraints_dir: Path,
) -> None:
    """Copy the base record's constraints npz to each scored record id."""
    source = (
        Path(source_work_dir) / "processed" / "constraints" / f"{base_record_id}.npz"
    )
    target_constraints_dir = Path(target_constraints_dir)
    target_constraints_dir.mkdir(parents=True, exist_ok=True)
    if not source.exists():
        logger.warning(
            "No constraints file at %s; skipping constraint linking.", source
        )
        return
    for record_id in record_ids:
        target = target_constraints_dir / f"{record_id}.npz"
        if not target.exists():
            shutil.copy(source, target)


def prepare_affinity_structure(
    complex_cif: Path,
    record_id: str,
    predictions_dir: Path,
    extra_mols_dir: Path,
    ccd: Optional[dict] = None,
    override: bool = False,
) -> Optional[Path]:
    """Parse a complex mmCIF into a Boltz ``pre_affinity_{record_id}.npz``.

    This mirrors Boltzina's ``_prepare_structure``: it parses the
    protein+ligand mmCIF and dumps the serialized ``StructureV2`` that the
    affinity data module loads.
    """
    from boltzina.data.parse.mmcif import parse_mmcif

    pose_dir = Path(predictions_dir) / record_id
    pose_dir.mkdir(parents=True, exist_ok=True)
    output_path = pose_dir / f"pre_affinity_{record_id}.npz"
    if output_path.exists() and not override:
        logger.info("Structure already prepared: %s", output_path)
        return pose_dir

    try:
        parsed = parse_mmcif(
            path=str(complex_cif),
            mols=ccd or {},
            moldir=str(extra_mols_dir),
            call_compute_interfaces=False,
        )
        parsed.data.dump(output_path)
        if not output_path.exists():
            raise RuntimeError(f"Failed to write {output_path}")
        return pose_dir
    except Exception as exc:  # noqa: BLE001
        logger.error("Failed to prepare structure %s: %s", complex_cif, exc)
        return None


def load_ccd(cache_dir: Optional[Path] = None, drop_name: Optional[str] = None) -> dict:
    """Load the Boltz CCD pickle (optionally dropping a residue name)."""
    cache_dir = resolve_cache_path(cache_dir)
    ccd_path = cache_dir / "ccd.pkl"
    if not ccd_path.exists():
        return {}
    with ccd_path.open("rb") as fh:
        ccd = pickle.load(fh)
    if drop_name and drop_name in ccd:
        ccd.pop(drop_name)
    return ccd


# --------------------------------------------------------------------------- #
# Results
# --------------------------------------------------------------------------- #
def extract_affinity_results(
    predictions_dir: Path,
    record_ids: List[str],
    extra: Optional[Dict[str, Any]] = None,
) -> List[dict]:
    """Collect the ``affinity_{record_id}.json`` outputs into a list of dicts."""
    results: List[dict] = []
    for record_id in record_ids:
        affinity_file = (
            Path(predictions_dir) / record_id / f"affinity_{record_id}.json"
        )
        if not affinity_file.exists():
            logger.warning("Missing affinity output for %s", record_id)
            continue
        with open(affinity_file) as fh:
            affinity = json.load(fh)
        affinity["record_id"] = record_id
        if extra:
            affinity.update(extra)
        results.append(affinity)
    return results
