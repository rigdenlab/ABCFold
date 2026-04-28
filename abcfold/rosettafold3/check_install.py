import logging
import urllib.request
from pathlib import Path

from abcfold.backend_envs import MicromambaEnv

logger = logging.getLogger("logger")

RF3_BASE_URL = "http://files.ipd.uw.edu/pub/rf3"
CHECKPOINT_NAME = "rf3_foundry_01_24_latest_remapped.ckpt"
RF3_URL = f"{RF3_BASE_URL}/{CHECKPOINT_NAME}"


def ensure_rosettafold_checkpoint(target_path: Path) -> Path:
    if target_path.exists():
        return target_path

    target_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        logger.info(
            "Downloading RoseTTAFold3 checkpoint via HTTPS "
            "(this may take a while)..."
        )
        urllib.request.urlretrieve(RF3_URL, target_path)
    except Exception as e:
        raise RuntimeError(
            "Failed to download RoseTTAFold3 checkpoint.\n"
            f"Target: {target_path}\n"
            f"Error: {e}"
        )

    if not target_path.exists():
        raise RuntimeError("Checkpoint download completed but file not found")

    return target_path


def ensure_rosettafold_env(config: dict) -> MicromambaEnv:
    ROSETTAFOLD_ENV = config['rosettafold_env']
    ROSETTAFOLD_VERSION = config['rosetta_version']

    env = MicromambaEnv(ROSETTAFOLD_ENV)
    # 1. Ensure env exists
    env.create(python_version="3.12")

    # 2. Check installed rosettafold version
    installed = env.get_installed_version("rc-foundry")

    if installed != ROSETTAFOLD_VERSION:
        if installed is None:
            logger.info("RosettaFold3 not found. Installing rc-foundry version: %s",
                        ROSETTAFOLD_VERSION)
        else:
            logger.info(
                "RosettaFold3 version mismatch (found %s). "
                "Installing correct version: %s",
                installed,
                ROSETTAFOLD_VERSION,
            )

        env.pip_install([
            f"rc-foundry[rf3]=={ROSETTAFOLD_VERSION}",
        ])
    else:
        logger.info("RosettaFold3 is already up-to-date (%s)", ROSETTAFOLD_ENV)

    # 3. Ensure runtime deps you *actually* need
    env.ensure_package("numpy")
    env.ensure_package("typer")
    env.ensure_package("matplotlib")

    return env
