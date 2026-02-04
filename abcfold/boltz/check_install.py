import logging

from abcfold.backend_envs import MicromambaEnv

logger = logging.getLogger("logger")


def ensure_boltz_env(config: dict) -> MicromambaEnv:
    BOLTZ_ENV = config['Environments']['boltz_env']
    BOLTZ_VERSION = config['Versions']['boltz_version']

    env = MicromambaEnv(BOLTZ_ENV)

    # 1. Ensure env exists
    env.create(python_version="3.11")

    # 2. Check installed boltz version
    installed = env.get_installed_version("boltz")

    if installed != BOLTZ_VERSION:
        if installed is None:
            logger.info("boltz not found. Installing version: %s", BOLTZ_ENV)
        else:
            logger.info(
                "boltz version mismatch (found %s). Installing correct version: %s",
                installed,
                BOLTZ_ENV,
            )
        env.pip_install([
            f"boltz=={BOLTZ_VERSION}",
            "cuequivariance_torch",
            "cuequivariance_ops_torch-cu12",
            "--no-cache-dir",
        ])
    else:
        logger.info("boltz is already up-to-date (%s)", BOLTZ_ENV)

    # 3. Ensure runtime deps you *actually* need
    env.ensure_package("numpy")
    env.ensure_package("typer")
    env.ensure_package("matplotlib")

    return env
