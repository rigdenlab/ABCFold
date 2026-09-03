import logging
from pathlib import Path

from abcfold.backend_envs import MicromambaEnv

logger = logging.getLogger("logger")


# Set requirment versions for chai dependencies
PANDERA_VERSION = "0.24.0"

# Default location for Chai-1's downloaded weights
DEFAULT_CHAI_DOWNLOADS_DIR = Path.home().joinpath(".chai1")


def resolve_chai_downloads_dir(config: dict) -> Path:
    """
    Resolve the directory Chai-1 should use for its downloaded weights, ESM
    embeddings, and conformer cache (i.e. what CHAI_DOWNLOADS_DIR should be
    set to).

    Args:
        config (dict): Configuration dictionary

    Returns:
        Path: Directory to pass to Chai-1 via CHAI_DOWNLOADS_DIR. Uses the
            'chai_weights' config option if set, otherwise ~/.chai1.
    """
    chai_weight_dir = config.get("chai_weights")
    if chai_weight_dir is not None and chai_weight_dir != "None":
        return Path(chai_weight_dir)
    return DEFAULT_CHAI_DOWNLOADS_DIR


def ensure_chai_env(config: dict) -> MicromambaEnv:
    CHAI_ENV = config['chai_env']
    CHAI_VERSION = config['chai_version']

    env = MicromambaEnv(CHAI_ENV)

    # 1. Ensure env exists
    env.create(python_version="3.11")

    # 2. Check installed chai version
    installed = env.get_installed_version("chai_lab")

    if installed != CHAI_VERSION:
        if installed is None:
            logger.info("chai_lab not found. Installing version: %s", CHAI_VERSION)
        else:
            logger.info(
                "chai_lab version mismatch (found %s). Installing correct version: %s",
                installed,
                CHAI_VERSION,
            )
        env.pip_install([f"chai_lab=={CHAI_VERSION}"])
    else:
        logger.info("chai_lab is already up-to-date (%s)", CHAI_VERSION)

    if not env.which("kalign"):
        logger.info("Installing kalign for Chai-1 template search")
        env.conda_install(["kalign"], channels=["conda-forge", "bioconda"])

    installed_pandera = env.get_installed_version("pandera")
    if installed_pandera != PANDERA_VERSION:
        logger.info("Installing pandera==%s for Chai compatibility", PANDERA_VERSION)
        env.pip_install([f"pandera=={PANDERA_VERSION}"])

    # 3. Ensure runtime deps you *actually* need
    env.ensure_package("numpy")
    env.ensure_package("typer")
    env.ensure_package("matplotlib")

    return env
