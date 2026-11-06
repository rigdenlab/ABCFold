import logging

from abcfold.backend_envs import MicromambaEnv

logger = logging.getLogger("logger")


def ensure_esmfold2_env(config: dict) -> MicromambaEnv:
    """
    Make sure the isolated ESMFold2 environment exists and holds the pinned
    version.

    ESMFold2 ships inside the ``esm`` package; weights are pulled from the
    ``biohub/ESMFold2`` HuggingFace repository on first use. Ligands given by
    CCD code additionally need ``ccd.pkl`` from the same repository, which the
    input builder downloads into its own cache.

    Args:
        config (dict): Configuration dictionary

    Returns:
        MicromambaEnv: The ESMFold2 environment
    """
    ESMFOLD2_ENV = config["esmfold2_env"]
    ESMFOLD2_VERSION = config["esmfold2_version"]

    env = MicromambaEnv(ESMFOLD2_ENV)
    # esm requires >=3.12
    env.create(python_version="3.12")

    installed = env.get_installed_version("esm")

    if installed != ESMFOLD2_VERSION:
        if installed is None:
            logger.info(
                "ESMFold2 not found. Installing esm version: %s", ESMFOLD2_VERSION
            )
        else:
            logger.info(
                "ESMFold2 version mismatch (found %s). Installing correct version: %s",
                installed,
                ESMFOLD2_VERSION,
            )

        env.pip_install([f"esm=={ESMFOLD2_VERSION}"])
    else:
        logger.info("ESMFold2 is already up-to-date (%s)", ESMFOLD2_ENV)

    return env
