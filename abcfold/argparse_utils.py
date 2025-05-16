import argparse
import logging
import sys
from pathlib import Path

logger = logging.getLogger("logger")


def validate_json_file(value):
    """
    Validate that the input is a JSON file with a .json suffix.
    """
    if not value.endswith(".json"):
        raise argparse.ArgumentTypeError(
            f"Input file must have a .json suffix: {value}"
        )
    if not Path(value).exists():
        raise argparse.ArgumentTypeError(f"Input file does not exist: {value}")
    return value


def main_argpase_util(parser):
    parser.add_argument(
        "input_json",
        type=validate_json_file,
        help="Path to the input JSON in AlphaFold3 format"
    )
    parser.add_argument(
        "output_dir",
        help="Path to the output directory"
    )
    parser.add_argument(
        "--override",
        help="[optional] Override the existing output directory, if it exists",
        action="store_true",
    )
    parser.add_argument(
        "--output_json",
        help="[optional] Specify the path of the output ABCFold json file, this \
can be used to run subsequent runs of ABCFold with the same input features (e.g. MSA)",
    )

    return parser


def mmseqs2_argparse_util(parser):
    parser.add_argument(
        "--mmseqs2",
        action="store_true",
        help="[optional] Use MMseqs2 for MSA generation and template \
searching (if used with --templates flag)",
    )
    parser.add_argument(
        "--mmseqs_database",
        help="[optional] The database directory for the generation of the MSA. This \
is only required if using a local installation of MMseqs2"
    )
    parser.add_argument(
        "--templates",
        action="store_true",
        help="[optional] Enable template search"
    )
    parser.add_argument(
        "--num_templates",
        type=int,
        default=20,
        help="[optional] The number of templates to use (default: 20)",
    )

    return parser


def custom_template_argpase_util(parser):
    parser.add_argument(
        "--target_id",
        nargs="+",
        help="[conditionally required] The ID of the sequence that the \
custom template relates to. This is only required if modelling a complex. \
If providing a list of custom templates, the target_id must be a list of \
the same length as the custom template list",
    )
    parser.add_argument(
        "--custom_template",
        nargs="+",
        help="[optional] Path to a custom template file in mmCif format or a list \
of paths to custom template files in mmCif format. If providing a list of \
custom templates, you must also provide a list of custom template chains.",
    )
    parser.add_argument(
        "--custom_template_chain",
        nargs="+",
        help="[conditionally required] The chain ID of the chain to use in your \
custom template. This is only required if using a multi-chain template. If \
providing a list of custom templates, you must also provide a list of custom \
template chains of the same length as the custom template list",
    )

    return parser


def prediction_argparse_util(parser):
    parser.add_argument(
        "--number_of_models",
        type=int,
        default=5,
        help="[optional] The number of models to generate with each method \
(default: 5)",
    )
    parser.add_argument(
        "--num_recycles",
        type=int,
        default=10,
        help="[optional] Number of recycles to use during inference (default: 10)",
    )
    return parser


def boltz_argparse_util(parser):
    parser.add_argument(
        "-b",
        "--boltz1",
        action="store_true",
        help="Run Boltz1",
    )
    if "--save_input" not in parser._option_string_actions:
        parser.add_argument(
            "--save_input",
            action="store_true",
            help="Save the input json file",
            default=False,
        )

    return parser


def chai_argparse_util(parser):
    parser.add_argument(
        "-c",
        "--chai1",
        action="store_true",
        help="Run Chai-1",
    )
    return parser


def alphafold_argparse_util(parser):
    parser.add_argument(
        "--database",
        help="[optional] The database directory for the generation of the MSA. This \
is only required if using the built in AlphaFold3 MSA generation",
        dest="database_dir",
        default=None,
    )

    parser.add_argument(
        "--model_params",
        help="[required] The directory containing the AlphaFold3 model parameters",
        default=None,
    )

    parser.add_argument(
        "--sif_path",
        help="[conditionally required] The path to the sif image of AlphaFold3 if \
using Singularity",
        default=None,
    )

    parser.add_argument(
        "-a",
        "--alphafold3",
        action="store_true",
        help="Run Alphafold3",
    )

    parser.add_argument(
        "--use_af3_template_search",
        action="store_true",
        help="If providing your own custom MSA or if you've run `--mmseqs2`, allow \
Alphafold3 to search for templates",
    )

    return parser


def visuals_argparse_util(parser):
    parser.add_argument(
        "--no_visuals",
        action="store_true",
        help="[optional] Do not generate the output pages, best for running on a \
cluster without a display",
    )

    parser.add_argument(
        "--no_server",
        action="store_true",
        help="[optional] Do not start a local server to view the results, the output \
page is still generated and is accessible in the output directory",
    )
    return parser


def raise_argument_errors(args):
    if not args.alphafold3 and not args.boltz1 and not args.chai1:
        logger.info(
            "Neither AlphaFold3, Boltz-1, or Chai-1 selected. Running AlphaFold3 \
by default"
        )
        args.alphafold3 = True

    if (
        args.alphafold3
        and (not args.model_params or not Path(args.model_params).exists())
        and not args.mmseqs2
    ):
        logger.error(f"Model parameters directory not found: {args.model_params}")
        sys.exit(1)

    if args.templates and not args.mmseqs2 and not args.alphafold3:
        logger.error(
            "Cannot use --templates flag without using MMseqs2 or Alphafold3"
        )
        sys.exit(1)

    if (
        args.templates
        and args.alphafold3
        and not args.mmseqs2
        and not args.use_af3_template_search
    ):
        # Ensure templates are used with Alphafold3 if --templates is set
        args.use_af3_template_search = True

    if args.custom_template_chain and not args.custom_template:
        logger.error("Custom template chain provided without a custom template")
        sys.exit(1)

    if args.use_af3_template_search and not args.alphafold3:
        logger.error(
            "Cannot use the Alphafold3 template search without running Alphafold3"
        )
        sys.exit(1)

    if args.num_templates < 1:
        logger.error("Number of templates must be greater than 0")
        sys.exit(1)

    if args.num_recycles < 1:
        logger.error("Number of recycles must be greater than 0")
        sys.exit(1)

    if args.number_of_models < 1:
        logger.error("Number of models must be greater than 0")
        sys.exit(1)

    return args
