import json
import logging
import os
import shutil
import time
from io import StringIO
from pathlib import Path
from typing import Mapping, Optional, Union

from Bio import Align
from Bio.PDB import MMCIFIO, MMCIFParser
from colorama import Fore, Style

logger = logging.getLogger("logger")


# Custom formatter for colored logging
class ColoredFormatter(logging.Formatter):
    # Define color codes for each log level
    LEVEL_COLORS = {
        logging.DEBUG: Fore.BLUE,
        logging.INFO: Fore.WHITE,
        logging.WARNING: Fore.YELLOW,
        logging.ERROR: Fore.RED,
        logging.CRITICAL: Fore.RED + Style.BRIGHT,
    }

    def format(self, record):
        # Get the color for the log level
        level_color = self.LEVEL_COLORS.get(record.levelno, "")
        # Format the log message
        formatted_message = super().format(record)
        # Return the message with the color added
        return f"{level_color}{formatted_message}{Style.RESET_ALL}"


# Set up logging
def setup_logger():
    logger = logging.getLogger("logger")
    logger.setLevel(logging.DEBUG)  # Set the minimum logging level

    # Create a stream handler (output to console)
    handler = logging.StreamHandler()
    handler.setLevel(logging.DEBUG)

    # Set the custom formatter
    formatter = ColoredFormatter("%(asctime)s - %(levelname)s - %(message)s")
    handler.setFormatter(formatter)

    # Add the handler to the logger
    logger.addHandler(handler)

    return logger


def get_chains(mmcif_file):
    """Return a list of chains in a MMCIF file."""
    parser = MMCIFParser(QUIET=True)
    structure = parser.get_structure("template", mmcif_file)
    chains = []
    for model in structure:
        for chain in model:
            chains.append(chain.id)
    return chains


def extract_sequence_from_mmcif(mmcif_file):
    """Extract the sequence from a MMCIF file."""
    parser = MMCIFParser(QUIET=True)
    structure = parser.get_structure("template", mmcif_file)
    sequence = ""
    model = structure[0]  # Assuming one model/chain only
    for chain in model:
        for residue in chain:
            if residue.id[0] == " ":  # Exclude heteroatoms
                sequence += residue.resname[0]  # Simplified to take the first letter
    return sequence


# Code from https://github.com/google-deepmind/alphafold3
def query_to_hit_mapping(
    query_aligned: str, template_aligned: str
) -> Mapping[int, int]:
    """0-based query index to hit index mapping."""
    query_to_hit_mapping_out = {}
    hit_index = 0
    query_index = 0
    for q_char, t_char in zip(query_aligned, template_aligned):
        # Gap inserted in the template
        if q_char == "-":
            query_index += 1
        # Deleted residue in the template (would be a gap in the query).
        elif t_char == "-":
            hit_index += 1
        # Normal aligned residue, in both query and template. Add to mapping.
        else:
            query_to_hit_mapping_out[query_index] = hit_index
            query_index += 1
            hit_index += 1
    return query_to_hit_mapping_out


def align_and_map(query_seq, template_seq):
    """Align two sequences and map the indices."""
    # Perform pairwise alignment
    aligner = Align.PairwiseAligner()
    alignments = aligner.align(query_seq, template_seq)
    alignment = alignments[0]  # Take the best alignment

    formatted_alignment = alignment._format_generalized().replace(" ", "")
    query_aligned, _, template_aligned, _ = formatted_alignment.split("\n")

    # Map the aligned sequences
    aligned_mapping = query_to_hit_mapping(query_aligned, template_aligned)

    query_indices = []
    template_indices = []
    for template_index, query_index in aligned_mapping.items():
        query_indices.append(query_index)
        template_indices.append(template_index)

    return query_indices, template_indices


def get_mmcif(
    cif,
    pdb_id,
    chain_id,
    start,
    end,
    tmpdir=None,
):
    """
    Extract a chain from a CIF file and return a new CIF string with only the
    specified chain, residues and metadata.
    """

    parser = MMCIFParser(QUIET=True)
    structure = parser.get_structure(pdb_id, cif)

    # Extract release date from the CIF file
    mmcif_dict = parser._mmcif_dict
    headers_to_keep = [
        "_entry.id",
        "_entry.title",
        "_entry.deposition_date",
        "_pdbx_audit_revision_history.revision_date",
    ]
    filtered_metadata = {
        key: mmcif_dict[key] for key in headers_to_keep if key in mmcif_dict
    }

    # Make metadata if missing
    if "_pdbx_audit_revision_history.revision_date" not in filtered_metadata:
        filtered_metadata["_pdbx_audit_revision_history.revision_date"] = time.strftime(
            "%Y-%m-%d"
        )

    # For multimodel templates (e.g. NMR) pick a single representative model
    # add a copy of the first model to the structure

    if len(structure) > 1:
        for model_index in range(1, len(structure)):
            structure.detach_child(structure[model_index].get_id())

    for model in structure:
        chain_to_del = []
        for chain in model:
            if chain.id != chain_id:
                chain_to_del.append(chain.id)
                continue

        for unwanted_chain in chain_to_del:
            model.detach_child(unwanted_chain)

        for chain in model:
            res_to_del = []
            for i, res in enumerate(chain):
                rel_pos = i + 1
                if rel_pos < start or rel_pos > end or res.id[0] != " ":
                    res_to_del.append(res)

            for res in res_to_del:
                chain.detach_child(res.id)

    # Save the filtered structure to a new CIF file
    io = MMCIFIO()
    io.set_structure(structure)
    filtered_output = (
        f"{pdb_id}_{chain_id}.cif"
        if tmpdir is None
        else f"{tmpdir}/{pdb_id}_{chain_id}.cif"
    )
    io.save(filtered_output)

    # Parse the filtered structure to get the modified MMCIF with no metadata
    structure = parser.get_structure(pdb_id, filtered_output)
    mmcif_dict = parser._mmcif_dict

    # Add the filtered metadata to the MMCIF dictionary
    mmcif_dict.update(filtered_metadata)

    # Save the modified MMCIF with wanted metadata to a string
    string_io = StringIO()
    io.set_dict(mmcif_dict)
    io.save(string_io)

    os.unlink(filtered_output)

    return string_io.getvalue()


# runs for each sequence in the input json
def get_custom_template(
    sequence,
    target_id,
    custom_template,
    custom_template_chain,
):
    """Add a custom template to the input json"""

    # add code here to run the custom template but move the res to output
    # Keep existing templates if they're present
    if "templates" not in sequence["protein"]:
        templates = []
    else:
        templates = sequence["protein"]["templates"]

    input_sequence = sequence["protein"]["sequence"]
    seq_id = sequence["protein"]["id"]
    if isinstance(seq_id, list):
        if target_id and target_id not in seq_id:
            return sequence
    if isinstance(seq_id, str):
        if target_id and target_id != seq_id:
            return sequence

    if not os.path.exists(custom_template):
        msg = f"Custom template file {custom_template} not found"
        logger.critical(msg)
        raise FileNotFoundError()

    chain_info = get_chains(custom_template)
    if len(chain_info) != 1 and not custom_template_chain:
        msg = f"Custom template file {custom_template} contains \
{len(chain_info)} chains. Please specify the chain to use with --custom_template_chain"
        raise ValueError(msg)

    if custom_template_chain and custom_template_chain not in chain_info:
        msg = f"Custom template file {custom_template} does not \
contain chain {custom_template_chain}"
        raise ValueError(msg)

    if not custom_template_chain:
        custom_template_chain = chain_info[0]

    template = {}
    cif_str = get_mmcif(
        custom_template,
        "custom",
        custom_template_chain,
        1,
        len(input_sequence),
    )

    template["mmcif"] = cif_str
    template_seq = extract_sequence_from_mmcif(StringIO(cif_str))
    query_indices, template_indices = align_and_map(input_sequence, template_seq)

    template["queryIndices"] = query_indices
    template["templateIndices"] = template_indices

    # Add the custom template to the start of the templates list
    templates.insert(0, template)

    # Add template to the json
    sequence["protein"]["templates"] = templates

    # Save the output json
    return sequence


def make_dir(dir_path: Union[str, Path], overwrite: bool = False):
    """
    Make a directory and return the Path object.

    Args:
        dir_path: The path to the directory to create.
        overwrite: Whether to delete the directory if it already exists.

    Returns:
        The Path object for the created directory.
    """
    dir_path = Path(dir_path)
    if dir_path.exists():
        if overwrite:
            shutil.rmtree(dir_path)
        else:
            logger.error(
                f"Directory {dir_path} already exists, use --override to replace it"
            )
            raise FileExistsError()

    dir_path.mkdir(parents=True, exist_ok=True)
    return dir_path


def check_input_json(
    input_json: Union[str, Path],
    output_dir: Optional[Union[str, Path]] = None,
    use_af3_templates: bool = False,
    test: bool = False,
):
    """
    Check the input json file for missing fields and add default values.

    Args:
        input_json: The path to the input json file.
        use_af3_templates: Whether to use the AlphaFold3 templates.

    Returns:
        None
    """
    input_json = Path(input_json)

    output_json = (
        input_json.parent.joinpath("abc_" + input_json.name)
        if output_dir is None
        else Path(output_dir).joinpath("abc_" + input_json.name)
    )
    with open(input_json, "r") as f:
        input_data = json.load(f)

    for sequence in input_data["sequences"]:
        for sequence_type in sequence:
            if (
                "unpairedMsa" in sequence[sequence_type]
                or "unpairedMsaPath" in sequence[sequence_type]
            ):
                if "templates" not in sequence[sequence_type]:

                    sequence[sequence_type]["templates"] = (
                        None if use_af3_templates else []
                    )

                if "pairedMsa" not in sequence[sequence_type]:
                    sequence[sequence_type]["pairedMsa"] = ""

    if test:
        return input_data
    with open(output_json, "w") as f:
        json.dump(input_data, f, indent=4)

    return output_json


def make_dummy_af3_db(output_dir):
    dummy_af3_db = output_dir.joinpath("af3_db")
    dummy_files = [
        "bfd-first_non_consensus_sequences.fasta",
        "rfam_14_9_clust_seq_id_90_cov_80_rep_seq.fasta",
        "uniprot_all_2021_04.fa",
        "mgy_clusters_2022_05.fa",
        "nt_rna_2023_02_23_clust_seq_id_90_cov_80_rep_seq.fasta",
        "pdb_seqres_2022_09_28.fasta",
        "rnacentral_active_seq_id_90_cov_80_linclust.fasta",
        "uniref90_2022_05.fa"
        ]
    dummy_dirs = ["mmcif_files"]

    dummy_af3_db.mkdir(parents=True, exist_ok=True)
    for dummy_file in dummy_files:
        with open(dummy_af3_db.joinpath(dummy_file), "w") as f:
            f.write("")
    for dummy_dir in dummy_dirs:
        dummy_af3_db.joinpath(dummy_dir).mkdir(parents=True, exist_ok=True)

    return dummy_af3_db
