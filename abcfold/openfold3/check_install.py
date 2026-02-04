import logging
import urllib.request
from pathlib import Path

from abcfold.backend_envs import MicromambaEnv

logger = logging.getLogger("logger")

OPENFOLD_BUCKET = "openfold"
CHECKPOINT_NAME = "of3_ft3_v1.pt"
S3_KEY = f"openfold3_params/{CHECKPOINT_NAME}"
HTTPS_URL = (
    f"https://{OPENFOLD_BUCKET}.s3.amazonaws.com/{S3_KEY}"
)


def ensure_openfold_checkpoint(target_path: Path) -> Path:
    """
    Ensure OpenFold3 checkpoint exists at target_path.
    Non-interactive, automation-safe.
    """
    if target_path.exists():
        return target_path

    target_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        logger.info(
            "Downloading OpenFold3 checkpoint via HTTPS "
            "(~2.1 GB, this may take a while)..."
        )
        urllib.request.urlretrieve(HTTPS_URL, target_path)
    except Exception as e:
        raise RuntimeError(
            "Failed to download OpenFold3 checkpoint.\n"
            f"Target: {target_path}\n"
            f"Error: {e}"
        )

    if not target_path.exists():
        raise RuntimeError("Checkpoint download completed but file not found")

    return target_path


def ensure_openfold_env(config: dict) -> MicromambaEnv:
    OPENFOLD_ENV = config['openfold_env']
    OPENFOLD_VERSION = config['openfold_version']

    env = MicromambaEnv(OPENFOLD_ENV)

    # 1. Ensure env exists
    env.create(python_version="3.11")

    # 2. Check installed openfold version
    installed = env.get_installed_version("openfold3")

    if installed != OPENFOLD_VERSION:
        if installed is None:
            logger.info("openfold3 not found. Installing version: %s", OPENFOLD_VERSION)
        else:
            logger.info(
                "openfold3 version mismatch (found %s). Installing correct version: %s",
                installed,
                OPENFOLD_VERSION,
            )
        # 3. Install Kalign2
        env.conda_install(["kalign2"], channels=["conda-forge", "bioconda"])

        env.pip_install([
            f"openfold3=={OPENFOLD_VERSION}",
            "cuequivariance_torch",
            "cuequivariance_ops_torch-cu12",
            "--no-cache-dir",
        ])
        # Setup databases
        env.run(["setup_openfold"])
    else:
        logger.info("openfold3 is already up-to-date (%s)", OPENFOLD_ENV)

    # 4. Ensure runtime deps you *actually* need
    env.ensure_package("numpy")
    env.ensure_package("typer")
    env.ensure_package("matplotlib")

    return env
