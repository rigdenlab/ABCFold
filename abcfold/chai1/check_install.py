import logging

from abcfold.backend_envs import MicromambaEnv

logger = logging.getLogger("logger")


# Set requirment versions for chai dependencies
PANDERA_VERSION = "0.24.0"


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
        env.conda_install(["kalign2"], channels=["conda-forge", "bioconda"])
        env.pip_install([f"chai_lab=={CHAI_VERSION}"])
    else:
        logger.info("chai_lab is already up-to-date (%s)", CHAI_VERSION)

    installed_pandera = env.get_installed_version("pandera")
    if installed_pandera != PANDERA_VERSION:
        logger.info("Installing pandera==%s for Chai compatibility", PANDERA_VERSION)
        env.pip_install([f"pandera=={PANDERA_VERSION}"])

    # 3. Ensure runtime deps you *actually* need
    env.ensure_package("numpy")
    env.ensure_package("typer")
    env.ensure_package("matplotlib")

    return env
