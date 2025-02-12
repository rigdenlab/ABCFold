import logging
import subprocess
import sys

logger = logging.getLogger("logger")


def check_chai1():
    try:
        import chai_lab as _  # noqa F40
    except (ImportError, ModuleNotFoundError):
        try:
            import boltz1 as _  # noqa F401

            no_deps = True
        except (ImportError, ModuleNotFoundError):
            no_deps = False
        logger.info("Installing chai_lab package")
        logger.info("No dependencies will be installed") if no_deps else None
        cmd = [sys.executable, "-m", "pip", "install", "chai_lab", "--no-cache-dir"]
        cmd.append("--no-deps") if no_deps else None
        logger.info("Running %s", " ".join(cmd))
        with subprocess.Popen(
            cmd,
            stdout=sys.stdout,
            stderr=subprocess.PIPE,
        ) as proc:
            proc.wait()
            if proc.returncode != 0:
                if proc.stderr:
                    logger.error(proc.stderr.read().decode())
                raise subprocess.CalledProcessError(proc.returncode, proc.args)

        # add pytroch lightning to install
