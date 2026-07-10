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
        from abcfold.chai1.check_install import ensure_chai_env
        from abcfold.chai1.run_chai1 import generate_chai_test_command

        env = ensure_chai_env(config=config)
        env.run(generate_chai_test_command(), quiet=True)

    if args.protenix:
        selected.append("Protenix")
        logger.info("[dry-run] Setting up Protenix environment...")
        from abcfold.protenix.check_install import ensure_protenix_env
        from abcfold.protenix.run_protenix import \
            generate_protenix_test_command

        env = ensure_protenix_env(config=config)
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
