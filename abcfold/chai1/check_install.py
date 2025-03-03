import logging
import subprocess
import sys

logger = logging.getLogger("logger")


CHAI_VERSION = "0.6.0"


def check_chai1():
    try:
        import chai_lab as _  # noqa F40

        cmd = [sys.executable, "-m", "pip", "show", "chai_lab"]
        with subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        ) as proc:
            stdout, stderr = proc.communicate()
            if proc.returncode != 0:
                raise subprocess.CalledProcessError(proc.returncode, cmd, stderr)

            version = None
            for line in stdout.decode().split("\n"):
                if line.startswith("Version:"):
                    version = line.split(":", 1)[1].strip()
                    break

            if version != CHAI_VERSION:
                raise ImportError(
                    f"Expected Chai-1 version {CHAI_VERSION}, found {version}"
                )
    except (ImportError, ModuleNotFoundError):
        try:
            import boltz1 as _  # noqa F401

            no_deps = True
        except (ImportError, ModuleNotFoundError):
            no_deps = False
        logger.info("Installing chai_lab package")
        logger.info("No dependencies will be installed") if no_deps else None
        cmd = [
            sys.executable,
            "-m",
            "pip",
            "install",
            f"chai_lab=={CHAI_VERSION}",
            "--no-cache-dir",
        ]
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
