import os
import shutil
import subprocess
import sys
from typing import Optional


class MicromambaEnv:
    def __init__(self, env_name: str):
        self.env_name = env_name
        self.micromamba = self._find_micromamba()

    @staticmethod
    def _find_micromamba() -> str:
        for candidate in [
            shutil.which("micromamba"),
            os.path.expanduser("~/micromamba/bin/micromamba"),
            os.path.expanduser("~/.local/bin/micromamba"),
        ]:
            if candidate and os.path.exists(candidate):
                return candidate
        raise RuntimeError(
            "micromamba binary not found. Ensure micromamba is installed and on PATH."
        )

    def get_installed_version(self, package: str) -> Optional[str]:
        """
        Return installed version of `package` inside the env,
        or None if the package is not installed.
        """
        code = (
            "from importlib.metadata import version, PackageNotFoundError\n"
            "import sys\n"
            f"pkg='{package}'\n"
            "try:\n"
            "    print(version(pkg))\n"
            "except PackageNotFoundError:\n"
            "    sys.exit(1)"
        )

        try:
            out = subprocess.check_output(
                [
                    self.micromamba,
                    "run",
                    "-n",
                    self.env_name,
                    "python",
                    "-c",
                    code,
                ],
                text=True,
            )
            return out.strip()
        except subprocess.CalledProcessError:
            return None

    def ensure_package(self, package: str, version: Optional[str] = None):
        """
        Ensure a package (optionally pinned to a version) is installed.
        """
        installed = self.get_installed_version(package)

        if installed is not None:
            if version is None or installed == version:
                return
            raise RuntimeError(
                f"{package}=={installed} is installed in {self.env_name}, "
                f"but {version} is required"
            )

        spec = f"{package}=={version}" if version else package
        self.pip_install([spec])

    def _run(self, args: list[str]):
        return subprocess.check_call([self.micromamba, *args])

    def env_exists(self) -> bool:
        out = subprocess.check_output(
            [self.micromamba, "env", "list"],
            text=True,
        )
        return any(
            line.split()[0] == self.env_name
            for line in out.splitlines()
            if line.strip()
        )

    def create(self, *, python_version: str):
        if self.env_exists():
            return
        self._run(
            [
                "create",
                "-y",
                "-n",
                self.env_name,
                f"python={python_version}",
            ]
        )

    def pip_install(self, packages: list[str]):
        self._run(
            [
                "run",
                "-n",
                self.env_name,
                "pip",
                "install",
                *packages,
            ]
        )

    def run(
        self, command: list[str], capture_output: bool = False, quiet: bool = False
    ) -> Optional[str]:
        cmd = [self.micromamba, "run", "-n", self.env_name, *command]

        if capture_output or quiet:
            stdout_lines = []
            with subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            ) as proc:
                if proc.stdout is None:
                    return None

                for line in proc.stdout:
                    if not quiet:
                        sys.stdout.write(line)
                        sys.stdout.flush()
                    stdout_lines.append(line)

                _, stderr = proc.communicate()

                if proc.returncode != 0:
                    if stderr and not quiet:
                        sys.stderr.write(stderr)
                        sys.stderr.flush()
                    raise subprocess.CalledProcessError(
                        proc.returncode,
                        cmd,
                        output="".join(stdout_lines),
                        stderr=stderr,
                    )

            return "".join(stdout_lines) if capture_output else None

        else:
            # Quiet is False and capture_output is False → just run
            subprocess.check_call(cmd)
            return None

    def remove(self):
        self._run(
            [
                "remove",
                "-n",
                self.env_name,
                "--all",
                "-y",
            ]
        )
