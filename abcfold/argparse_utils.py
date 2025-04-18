import logging
import sys
from pathlib import Path

logger = logging.getLogger("logger")


def main_argpase_util(parser):
    parser.add_argument("input_json", help="Input sequence file")
    parser.add_argument("output_dir", help="Output directory")
    parser.add_argument(
        "--gpus",
        type=str,
        default="all",
        help="GPU device(s) to use, e.g. 'all', '0', '0,1', 'cpu'"
    )

    return parser


def mmseqs2_argparse_util(parser):
    parser.add_argument(
        "--mmseqs2",
        action="store_true",
        help="Use MMseqs2 for MSA",
    )
    parser.add_argument(
        "--templates", action="store_true", help="Include templates in the output json"
    )
    parser.add_argument(
        "--num_templates",
        type=int,
        default=20,
        help="Number of templates to include in the output json",
    )

    return parser


def custom_template_argpase_util(parser):
    parser.add_argument(
        "--target_id", nargs="+", help="Target id relating to the custom template"
    )
    parser.add_argument(
        "--custom_template",
        nargs="+",
        help="Custom template to include in the output json",
    )
    parser.add_argument(
        "--custom_template_chain",
        nargs="+",
        help="Custom template chain to include in the output json",
    )

    return parser


def prediction_argparse_util(parser):
    parser.add_argument(
        "--number_of_models",
        type=int,
        default=5,
        help="Number of models to generate",
    )
    parser.add_argument(
        "--num_recycles",
        type=int,
        default=10,
        help="Number of recycles to use during the inference",
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

    parser.add_argument("--output_json", help="Output json file")
    # make the vartible saved as database_dir
    parser.add_argument(
        "--database",
        help="The Database directory for the generation of the MSA.",
        dest="database_dir",
        default=None,
    )

    parser.add_argument(
        "--model_params",
        help="The directory containing the model parameters",
        default=None,
    )

    parser.add_argument(
        "-a",
        "--alphafold3",
        action="store_true",
        help="Run Alphafold3",
    )

    parser.add_argument(
        "--override",
        help="Override the existing output directory, if it exists",
        action="store_true",
    )

    parser.add_argument(
        "--use_af3_template_search",
        action="store_true",
        help="If providing your own custom MSA or you've ran `--mmseqs`, allow \
Alphafold3 to search for templates",
    )

    return parser


def visuals_argparse_util(parser):
    parser.add_argument(
        "--no_visuals",
        action="store_true",
        help="Do not generate the output pages, best for running on a cluster\
 without a display",
    )
    parser.add_argument(
        "--no_server",
        action="store_true",
        help="Do not start a local server to view the results, the output page is \
stil generated and is accessible in the output directory",
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

    if args.templates and not args.mmseqs2:
        logger.error("Cannot include templates without using MMseqs2")
        sys.exit(1)

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
