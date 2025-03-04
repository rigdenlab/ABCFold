#!/usr/bin/env python

import json
import logging
import os

from abcfold.abc_script_utils import get_custom_template
from abcfold.argparse_utils import custom_template_argpase_util

logger = logging.getLogger("logger")


def run_custom_template(
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

        for template in custom_template:
            if not os.path.exists(template):
                msg = f"Custom template file {template} not found"
                logger.critical(msg)
                raise FileNotFoundError()
            # Can only add templates to protein sequences, so check if there
            # are multiple protein sequences in the input json
            if (
                len(
                    [
                        x
                        for x in af3_json["sequences"]
                        if "protein" in x.keys()
                    ]
                )
                > 1
                and not target_id
            ):
                msg = "Multiple sequences found in input json. \
Please specify target id so that custom template can be added to the correct sequence"
                raise ValueError(msg)

        if target_id and len(target_id) > 1:
            if (len(custom_template) != len(target_id)) or (
                len(custom_template_chain) != len(target_id)
            ):
                msg = "If providing templates for multiple targets, the number of \
target ids must match the number of custom templates and custom template chains"
                raise ValueError(msg)
            custom_templates = zip(
                target_id, custom_template, custom_template_chain
                )
        else:
            if len(custom_template) != len(custom_template_chain):
                msg = "Number of custom templates must match the number of \
custom template chains"
                raise ValueError(msg)
            # if a single target id is provided, assume all custom templates
            # are for the same target
            if target_id:
                target_ids = [target_id[0]] * len(custom_template)
            else:
                target_ids = [None] * len(custom_template)
            custom_templates = zip(
                target_ids, custom_template, custom_template_chain
                )

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

    run_custom_template(  # pragma: no cover
        args.input_json,
        args.target_id,
        args.custom_template,
        args.custom_template_chain,
        output_json=args.output_json,
        to_file=True,
    )


if __name__ == "__main__":  # pragma: no cover
    main()
