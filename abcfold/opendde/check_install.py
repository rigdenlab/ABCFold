import logging

from abcfold.backend_envs import MicromambaEnv

logger = logging.getLogger("logger")


def ensure_opendde_env(config: dict) -> MicromambaEnv:
    """
    Make sure the isolated OpenDDE environment exists and holds the pinned
    version.

    OpenDDE is a preview release: its CLI flags, JSON fields and checkpoints may
    change between versions, so the version is pinned in the config rather than
    tracking the latest release.

    Args:
        config (dict): Configuration dictionary

    Returns:
        MicromambaEnv: The OpenDDE environment
    """
    OPENDDE_ENV = config["opendde_env"]
    OPENDDE_VERSION = config["opendde_version"]

    env = MicromambaEnv(OPENDDE_ENV)
    # OpenDDE requires >=3.11,<3.14
    env.create(python_version="3.11")

    installed = env.get_installed_version("opendde")

    if installed != OPENDDE_VERSION:
        if installed is None:
            logger.info(
                "OpenDDE not found. Installing opendde version: %s", OPENDDE_VERSION
            )
        else:
            logger.info(
                "OpenDDE version mismatch (found %s). Installing correct version: %s",
                installed,
                OPENDDE_VERSION,
            )

        env.pip_install([f"opendde[gpu]=={OPENDDE_VERSION}"])
    else:
        logger.info("OpenDDE is already up-to-date (%s)", OPENDDE_ENV)

    return env
