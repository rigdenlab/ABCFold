import logging

from abcfold.backend_envs import MicromambaEnv

logger = logging.getLogger("logger")


def ensure_intellifold_env(config: dict) -> MicromambaEnv:
    """
    Make sure the isolated IntelliFold environment exists and holds the pinned
    version.

    The ``intellifold`` wheel is a thin wrapper around a vendored AlphaFold 3
    JAX engine. The engine itself is not on PyPI: it is built from the
    IntelliFold checkout with ``pip install ./third_party/alphafold3 &&
    build_data``. If the engine is missing the CLI exits with
    "the AlphaFold 3 JAX engine (module 'alphafold3') is not installed", so
    that case is reported with the build instructions rather than silently
    failing mid-run.

    Args:
        config (dict): Configuration dictionary

    Returns:
        MicromambaEnv: The IntelliFold environment
    """
    INTELLIFOLD_ENV = config["intellifold_env"]
    INTELLIFOLD_VERSION = config["intellifold_version"]

    env = MicromambaEnv(INTELLIFOLD_ENV)
    # IntelliFold v2 requires >=3.12
    env.create(python_version="3.12")

    installed = env.get_installed_version("intellifold")

    if installed != INTELLIFOLD_VERSION:
        if installed is None:
            logger.info(
                "IntelliFold not found. Installing intellifold version: %s",
                INTELLIFOLD_VERSION,
            )
        else:
            logger.info(
                "IntelliFold version mismatch (found %s). "
                "Installing correct version: %s",
                installed,
                INTELLIFOLD_VERSION,
            )

        env.pip_install([f"intellifold=={INTELLIFOLD_VERSION}"])
    else:
        logger.info("IntelliFold is already up-to-date (%s)", INTELLIFOLD_ENV)

    if env.get_installed_version("alphafold3") is None:
        raise RuntimeError(
            "IntelliFold needs the vendored AlphaFold 3 JAX engine, which is "
            "not distributed on PyPI. Build it into the "
            f"{INTELLIFOLD_ENV} environment with:\n"
            "  git clone https://github.com/IntelliGen-AI/IntelliFold\n"
            "  cd IntelliFold\n"
            "  pip install ./third_party/alphafold3 && build_data\n"
            "Building the engine needs a C++ toolchain (cmake/ninja/compiler) "
            "and CUDA, and pulls jax[cuda12]==0.9.1."
        )

    return env
