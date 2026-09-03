import contextlib
import logging
import socket
import urllib.request
from pathlib import Path
from typing import Optional

from abcfold.backend_envs import MicromambaEnv

logger = logging.getLogger("logger")

# Try the official protenix checkpoint source first,
# but fall back to a HuggingFace mirror if that fails
# since the official link is not always reachable.
OFFICIAL_BASE_URL = "https://protenix.tos-cn-beijing.volces.com/checkpoint"
HF_MIRROR_BASE_URL = (
    "https://huggingface.co/TMF001/pxdesign-weights/resolve/main/checkpoint"
)
OFFICIAL_TIMEOUT_SECONDS = 20


@contextlib.contextmanager
def _socket_timeout(seconds):
    previous = socket.getdefaulttimeout()
    socket.setdefaulttimeout(seconds)
    try:
        yield
    finally:
        socket.setdefaulttimeout(previous)


def _download(url: str, dest: Path, timeout: Optional[int] = None) -> None:
    if timeout is not None:
        with _socket_timeout(timeout):
            urllib.request.urlretrieve(url, dest)
    else:
        urllib.request.urlretrieve(url, dest)


def ensure_protenix_checkpoint(checkpoint_dir: Path, model_name: str) -> Path:
    """
    Ensure the Protenix checkpoint for `model_name` exists in `checkpoint_dir`.

    Protenix (via ``runner.inference``) expects to find the checkpoint at
    ``<checkpoint_dir>/<model_name>.pt``. If it isn't there, this tries to
    download it from Protenix's official source first, then falls back to a
    HuggingFace mirror (https://huggingface.co/TMF001/pxdesign-weights) if
    that fails, since the official link is not always reachable.

    Args:
        checkpoint_dir (Path): Directory that should contain (or will receive)
            the checkpoint file. This is what should be passed to Protenix as
            ``--load_checkpoint_dir``.
        model_name (str): Protenix model name (e.g. "protenix-v2").

    Returns:
        Path: Path to the checkpoint file.

    Raises:
        RuntimeError: If the checkpoint could not be downloaded from either
            source.
    """
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_path = checkpoint_dir.joinpath(f"{model_name}.pt")

    if checkpoint_path.exists():
        return checkpoint_path

    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    official_url = f"{OFFICIAL_BASE_URL}/{model_name}.pt"
    mirror_url = f"{HF_MIRROR_BASE_URL}/{model_name}.pt"

    try:
        logger.info(
            "Downloading Protenix checkpoint '%s' from the official source "
            "(this may take a while):\n%s -> %s",
            model_name, official_url, checkpoint_path,
        )
        _download(official_url, checkpoint_path, timeout=OFFICIAL_TIMEOUT_SECONDS)
        return checkpoint_path
    except Exception as e:
        if checkpoint_path.exists():
            checkpoint_path.unlink()
        logger.warning(
            "Could not reach Protenix's official checkpoint source (%s). "
            "Falling back to the HuggingFace mirror.",
            e,
        )

    try:
        logger.info(
            "Downloading Protenix checkpoint '%s' from the HuggingFace mirror "
            "(https://huggingface.co/TMF001/pxdesign-weights) instead:\n%s -> %s",
            model_name, mirror_url, checkpoint_path,
        )
        _download(mirror_url, checkpoint_path)
    except Exception as e:
        if checkpoint_path.exists():
            checkpoint_path.unlink()
        raise RuntimeError(
            f"Failed to download the Protenix checkpoint '{model_name}' from "
            "either the official source or the HuggingFace mirror.\n"
            f"Target: {checkpoint_path}\n"
            f"Official URL: {official_url}\n"
            f"Mirror URL: {mirror_url}\n"
            f"Error: {e}\n\n"
            "If the checkpoint isn't available at either location (e.g. for a "
            "custom or newer model), download it manually and place it at the "
            "path above, or point the 'protenix_weights' option in "
            "~/.abcfold_config.ini at a directory that already contains it."
        )

    if not checkpoint_path.exists():
        raise RuntimeError("Checkpoint download completed but file not found")

    return checkpoint_path


def ensure_protenix_env(config: dict) -> MicromambaEnv:
    PROTENIX_ENV = config['protenix_env']
    PROTENIX_VERSION = config['protenix_version']

    env = MicromambaEnv(PROTENIX_ENV)

    # 1. Ensure env exists
    env.create(python_version="3.11")

    # 2. Check installed protenix version
    installed = env.get_installed_version("protenix")

    if installed != PROTENIX_VERSION:
        if installed is None:
            logger.info("Protenix not found. Installing version: %s", PROTENIX_ENV)
        else:
            logger.info(
                "Protenix version mismatch (found %s). Installing correct version: %s",
                installed,
                PROTENIX_VERSION,
            )
        env.pip_install([f"protenix=={PROTENIX_VERSION}"])
    else:
        logger.info("Protenix is already up-to-date (%s)", PROTENIX_ENV)

    # 3. Ensure runtime deps you *actually* need
    env.ensure_package("numpy")
    env.ensure_package("typer")
    env.ensure_package("matplotlib")

    return env
