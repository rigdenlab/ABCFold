#!/usr/bin/env python

import json
import logging
from pathlib import Path

from Bio.PDB import MMCIFIO, MMCIFParser, PDBParser

from abcfold.argparse_utils import custom_template_argpase_util
from abcfold.scripts.abc_script_utils import get_custom_template

logger = logging.getLogger("logger")


def add_custom_template(
    input_json,
    target_id,
    custom_template,
    custom_template_chain,
    output_json=None,
    to_file=True,
):
    af3_json = json.load(open(input_json))

    for sequence in af3_json["sequences"]:
        if "protein" not in sequence:
            continue

        checked_templates = []
        for template in custom_template:
            if not Path(template).exists():
                msg = f"Custom template file {template} not found"
                logger.critical(msg)
                raise FileNotFoundError()

            # convert input pdb files to cif files
            if Path(template).suffix == ".pdb":
                pdb_parser = PDBParser(QUIET=True)
                structure = pdb_parser.get_structure(Path(template).stem, template)
                io = MMCIFIO()
                io.set_structure(structure)
                cif_template = Path(template).with_suffix(".cif")
                io.save(str(cif_template))
                template = cif_template

            # Set custom chain ID to the first chain if not specified
            if custom_template_chain is None:
                parser = MMCIFParser(QUIET=True)
                structure = parser.get_structure("custom", template)
                for model in structure:
                    for chain in model:
                        custom_template_chain = chain.id
                        break

            # Can only add templates to protein sequences, so check if there
            # are multiple protein sequences in the input json
            if (
                len([x for x in af3_json["sequences"] if "protein" in x.keys()]) > 1
                and not target_id
            ):
                msg = "Multiple sequences found in input json. \
Please specify target id so that custom template can be added to the correct sequence"
                raise ValueError(msg)
            checked_templates.append(template)

        if target_id and len(target_id) > 1:
            if (len(checked_templates) != len(target_id)) or (
                len(custom_template_chain) != len(target_id)
            ):
                msg = "If providing templates for multiple targets, the number of \
target ids must match the number of custom templates and custom template chains"
                raise ValueError(msg)
            custom_templates = zip(target_id, checked_templates, custom_template_chain)
        else:
            if len(checked_templates) != len(custom_template_chain):
                msg = "Number of custom templates must match the number of \
custom template chains"
                raise ValueError(msg)
            # if a single target id is provided, assume all custom templates
            # are for the same target
            if target_id:
                target_ids = [target_id[0]] * len(checked_templates)
            else:
                target_ids = [None] * len(checked_templates)
            custom_templates = zip(target_ids, checked_templates, custom_template_chain)

        for i in custom_templates:
            tid, c_tem, c_tem_chn = i
            sequence = get_custom_template(
                sequence,
                tid,
                c_tem,
                c_tem_chn,
            )

    if to_file:
        if not output_json:
            output_json = input_json

        with open(output_json, "w") as f:
            json.dump(af3_json, f)

    return af3_json


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Add custom template to alphafold input JSON"
    )

    parser.add_argument("--input_json", help="Input alphafold3 json file")
    parser.add_argument("--output_json", help="Output alphafold3 json file")
    parser = custom_template_argpase_util(parser)

    args = parser.parse_args()

    add_custom_template(  # pragma: no cover
        args.input_json,
        args.target_id,
        args.custom_template,
        args.custom_template_chain,
        output_json=args.output_json,
        to_file=True,
    )


if __name__ == "__main__":  # pragma: no cover
    main()
