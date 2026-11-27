"""Dry-run / setup-only support for ABCFold.

Creates the micromamba environments, downloads model weights, and runs each
predictor's ``--help`` smoke test -- but never runs inference. This lets a
central installation (e.g. via Ansible) be primed on a machine *without* GPUs,
ready for real runs elsewhere.

Only the predictors selected on the command line are set up. Nothing here
touches CUDA: the packages install their CUDA wheels, but the only command run
is ``<tool> ... --help`` (or, for AlphaFold3, the existing image/version check).
"""

import logging
from pathlib import Path

logger = logging.getLogger("logger")


def _weight_dir(config: dict, key: str, default: Path) -> Path:
    """Resolve a configured weights dir, falling back to a default path."""
    value = config.get(key)
    if value and value != "None":
        return Path(value)
    return default


def setup_environments(args, config: dict) -> None:
    """Set up (env + weights + smoke test) each selected predictor. No inference.

    Args:
        args (argparse.Namespace): Parsed CLI arguments (predictor flags).
        config (dict): Flattened runtime config (weights dirs, versions, envs).

    Raises:
        Exception: Propagates any setup/smoke-test failure so the process exits
            non-zero (Ansible and other automation can detect a bad install).
    """
    selected = []

    if args.alphafold3:
        selected.append("AlphaFold3")
        logger.info("[dry-run] Verifying AlphaFold3 image / parameters...")
        from abcfold.alphafold3.check_install import check_af3_install

        check_af3_install(
            config=config, interactive=False, sif_path=args.af3_sif_path
        )

    if args.boltz:
        selected.append("Boltz")
        logger.info("[dry-run] Setting up Boltz environment...")
        from abcfold.boltz.check_install import ensure_boltz_env
        from abcfold.boltz.run_boltz import generate_boltz_test_command

        env = ensure_boltz_env(config=config)
        env.run(generate_boltz_test_command(), quiet=True)

    if args.chai1:
        selected.append("Chai-1")
        logger.info("[dry-run] Setting up Chai-1 environment...")
        import os

        from abcfold.chai1.check_install import (ensure_chai_env,
                                                 resolve_chai_downloads_dir)
        from abcfold.chai1.run_chai1 import generate_chai_test_command

        env = ensure_chai_env(config=config)
        chai_downloads_dir = resolve_chai_downloads_dir(config)
        chai_downloads_dir.mkdir(parents=True, exist_ok=True)
        os.environ["CHAI_DOWNLOADS_DIR"] = str(chai_downloads_dir)
        env.run(generate_chai_test_command(), quiet=True)

    if args.protenix:
        selected.append("Protenix")
        logger.info("[dry-run] Setting up Protenix environment + checkpoint...")
        from abcfold.protenix.check_install import (ensure_protenix_checkpoint,
                                                    ensure_protenix_env)
        from abcfold.protenix.run_protenix import \
            generate_protenix_test_command

        env = ensure_protenix_env(config=config)
        cache_path = _weight_dir(
            config, "protenix_weights", Path.home().joinpath("checkpoint")
        )
        ensure_protenix_checkpoint(cache_path, config["protenix_model"])
        env.run(generate_protenix_test_command(), quiet=True)

    if args.openfold3:
        selected.append("OpenFold3")
        logger.info("[dry-run] Setting up OpenFold3 environment + checkpoint...")
        from abcfold.openfold3.check_install import (
            CHECKPOINT_NAME, ensure_openfold_checkpoint, ensure_openfold_env)
        from abcfold.openfold3.run_openfold3 import \
            generate_openfold_test_command

        env = ensure_openfold_env(config=config)
        cache_path = _weight_dir(
            config, "openfold_weights", Path.home().joinpath(".openfold3")
        )
        ensure_openfold_checkpoint(cache_path.joinpath(CHECKPOINT_NAME))
        env.run(generate_openfold_test_command(), quiet=True)

    if args.rosettafold3:
        selected.append("RoseTTAFold3")
        logger.info(
            "[dry-run] Setting up RoseTTAFold3 environment + checkpoint..."
        )
        from abcfold.rosettafold3.check_install import (
            CHECKPOINT_NAME, ensure_rosettafold_checkpoint,
            ensure_rosettafold_env)
        from abcfold.rosettafold3.run_rosettafold3 import \
            generate_rosettafold_test_command

        env = ensure_rosettafold_env(config=config)
        cache_path = _weight_dir(
            config, "rosettafold_weights", Path.home().joinpath(".rosettafold3")
        )
        ensure_rosettafold_checkpoint(cache_path.joinpath(CHECKPOINT_NAME))
        env.run(generate_rosettafold_test_command(), quiet=True)

    if not selected:
        logger.warning(
            "[dry-run] No predictors selected -- nothing to set up. Pass one or "
            "more of -a/-b/-c/-p/-o/-r to choose which environments to build."
        )
        return

    logger.info(
        "[dry-run] Setup complete for: %s. Environments and weights are ready; "
        "no inference was run.",
        ", ".join(selected),
    )


def main() -> None:
    """Standalone entry point: set up selected predictor environments.

    Run directly with e.g.::

        python -m abcfold.dry_run -b -c
        python -m abcfold.dry_run -abcopr

    Loads the same config file as ``abcfold`` (``~/.abcfold_config.ini``,
    created from the packaged defaults on first use) and sets up only the
    predictors chosen with -a/-b/-c/-p/-o/-r. No input JSON or GPU is required.
    """
    import argparse
    import configparser
    import shutil

    from abcfold.argparse_utils import (alphafold_argparse_util,
                                        boltz_argparse_util,
                                        chai_argparse_util,
                                        openfold_argparse_util,
                                        protenix_argparse_util,
                                        rosettafold_argparse_util)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument(
        "--config-file",
        type=str,
        default=str(Path.home() / ".abcfold_config.ini"),
        help="Path to the config file (defaults to ~/.abcfold_config.ini).",
    )
    config_args, _ = config_parser.parse_known_args()

    config_file = Path(config_args.config_file)
    default_config_file = Path(__file__).parent.joinpath("data", "config.ini")
    if not config_file.exists():
        shutil.copy(default_config_file, config_file)

    config = configparser.ConfigParser()
    config.read(str(config_file))
    rt_config: dict = {}
    for section in config.sections():
        rt_config.update(dict(config.items(section)))

    parser = argparse.ArgumentParser(
        description=(
            "Set up ABCFold predictor environments (env + weights + --help "
            "smoke test) without running inference. No GPU required."
        ),
        parents=[config_parser],
    )
    parser = alphafold_argparse_util(parser)
    parser = boltz_argparse_util(parser)
    parser = chai_argparse_util(parser)
    parser = openfold_argparse_util(parser)
    parser = protenix_argparse_util(parser)
    parser = rosettafold_argparse_util(parser)
    args = parser.parse_args()

    setup_environments(args, rt_config)


if __name__ == "__main__":
    main()
