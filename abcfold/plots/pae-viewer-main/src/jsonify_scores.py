#!/usr/bin/env python

"""
usage: jsonify_scores.py [-h] [-o OUTPUT_PATH] input_path

Unpickles the output of an AlphaFold-Multimer run and converts it into a
`.json` file which can be used by PAE Viewer.

Warning: The pickle module is not secure. Only unpickle data you trust. It is
possible to construct malicious pickle data which will execute arbitrary code
during unpickling. Never unpickle data that could have come from an untrusted
source, or that could have been tampered with.

positional arguments:
  input_path            Path to the `.pickle` file containing scores as
                        generated by AlphaFold.

options:
  -h, --help            show this help message and exit
  -o OUTPUT_PATH, --output_path OUTPUT_PATH
                        Path to the output JSON file. Will fail if file
                        already exists. Saves the output in the same directory
                        as the input file by default.
"""
import argparse
import json
import pickle
import sys
from pathlib import Path

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description=(
            "Unpickles the output of an AlphaFold-Multimer run and converts it"
            " into a `.json` file which can be used by PAE Viewer. Warning: The"
            " pickle module is not secure. Only unpickle data you trust. It is"
            " possible to construct malicious pickle data which will execute"
            " arbitrary code during unpickling. Never unpickle data that could"
            " have come from an untrusted source, or that could have been"
            " tampered with."
        )
    )

    parser.add_argument(
        'input_path',
        help=(
            "Path to the `.pickle` file containing scores as generated by"
            " AlphaFold."
        ),
    )

    parser.add_argument(
        '-o',
        '--output_path',
        help=(
            "Path to the output JSON file. Will fail if file already exists."
            " Saves the output in the same directory as the input file by"
            " default."
        ),
    )

    args = parser.parse_args()

    input_path = Path(args.input_path).resolve()
    output_path = args.output_path

    if output_path:
        output_path = Path(output_path).resolve()
    else:
        output_path = input_path.with_suffix('.json')

    if output_path.exists():
        sys.exit(f"File under output path '{output_path}' already exists!")

    with (
        open(input_path, 'rb') as input_file,
        open(output_path, 'w') as output_file,
    ):
        scores = pickle.load(input_file)

        output_scores = {
            'pae': scores['predicted_aligned_error'].tolist()
        }

        try:
            output_scores['max_pae'] = scores[
                'max_predicted_aligned_error'
            ].item()
        except KeyError:
            pass

        try:
            output_scores['plddt'] = scores['plddt'].tolist()
        except KeyError:
            pass

        try:
            output_scores['ptm'] = scores['ptm'].item()
        except KeyError:
            pass

        try:
            output_scores['iptm'] = scores['iptm'].item()
        except KeyError:
            pass

        json.dump(output_scores, output_file, ensure_ascii=False)
