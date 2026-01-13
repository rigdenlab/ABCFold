import os
from pathlib import Path

import nox

try:
    import tomllib
except ModuleNotFoundError or ImportError:
    import tomli as tomllib


def get_python_version():
    """Get the python version from .python-version file unless it is github actions."""
    if "GITHUB_ACTIONS" in os.environ:
        return Path(os.environ["Python_ROOT_DIR"]).parent.name
    return Path(".python-version").read_text().strip()


def get_package_name():
    """Get the package name from pyproject.toml."""
    pyproject_path = Path("pyproject.toml")
    pyproject_data = tomllib.loads(pyproject_path.read_text())
    return pyproject_data["project"]["name"].replace("-", "_")


PYTHON_VERSION = get_python_version()
PACKAGE_NAME = get_package_name()

# Default: reuse uv-managed virtualenvs for speed
nox.options.reuse_existing_virtualenvs = True


def uv(session, *args):
    """Run uv commands inside a nox session."""
    session.run("uv", *args, "--active", external=True)


@nox.session(python=PYTHON_VERSION)
def install(session):
    """
    Equivalent of:
      make install
    """
    uv(session, "sync", "--group", "dev")


@nox.session(python=PYTHON_VERSION)
def lock(session):
    """
    Equivalent of:
      make lock
    """
    uv(session, "lock", "--upgrade")


@nox.session(python=PYTHON_VERSION)
def lock_check(session):
    """
    Equivalent of:
      make lock-check
    """
    uv(session, "lock", "--check")


#
# Formatting / chores
#


@nox.session(python=PYTHON_VERSION)
def ruff_fixes(session):
    uv(session, "sync", "--group", "dev")
    session.run("ruff", "check", ".", "--fix")


@nox.session(python=PYTHON_VERSION)
def black_fixes(session):
    uv(session, "sync", "--group", "dev")
    session.run("ruff", "format", ".")


@nox.session(python=PYTHON_VERSION)
def tomlsort_fixes(session):
    uv(session, "sync", "--group", "dev")
    session.run(
        "tombi",
        "format",
        *Path(".").rglob("*.toml"),
        external=True,
    )


@nox.session(python=PYTHON_VERSION)
def isort_fixes(session):
    uv(session, "sync", "--group", "dev")
    session.run("isort", PACKAGE_NAME, "tests")


@nox.session(python=PYTHON_VERSION)
def chores(session):
    """
    Equivalent of:
      make chores
    """
    session.notify("isort_fixes")
    session.notify("ruff_fixes")
    session.notify("black_fixes")
    session.notify("tomlsort_fixes")


#
# Checks / linting
#


@nox.session(python=PYTHON_VERSION)
def ruff_check(session):
    uv(session, "sync", "--group", "dev")
    session.run("ruff", "check")


@nox.session(python=PYTHON_VERSION)
def black_check(session):
    uv(session, "sync", "--group", "dev")
    session.run("ruff", "format", ".", "--check")


@nox.session(python=PYTHON_VERSION)
def mypy_check(session):
    uv(session, "sync", "--group", "dev")
    session.run("mypy", PACKAGE_NAME)


@nox.session(python=PYTHON_VERSION)
def tomlsort_check(session):
    uv(session, "sync", "--group", "dev")
    tomls = [str(p) for p in Path(".").rglob("*.toml") if ".venv" not in p.parts]
    session.run("tombi", "lint", *tomls, external=True)
    session.run("tombi", "format", *tomls, "--check", external=True)


@nox.session(python=PYTHON_VERSION)
def isort_check(session):
    uv(session, "sync", "--group", "dev")
    session.run("isort", PACKAGE_NAME, "tests", "--check-only")


#
# Testing
#


@nox.session(python=PYTHON_VERSION)
def tests(session):
    """
    Equivalent of:
      make tests
    """

    session.notify("ruff_check")
    session.notify("black_check")
    session.notify("mypy_check")
    session.notify("tomlsort_check")
    session.notify("isort_check")
    session.notify("pytest")


@nox.session(python=PYTHON_VERSION)
def pytest(session):
    uv(session, "sync", "--group", "dev")
    session.run(
        "pytest",
        f"--cov={PACKAGE_NAME}",
        "--cov-report=term-missing",
        "tests",
    )


@nox.session(python=PYTHON_VERSION)
def pytest_loud(session):
    uv(session, "sync", "--group", "dev")
    session.run(
        "pytest",
        "--log-cli-level=DEBUG",
        "-log_cli=true",
        f"--cov=./{PACKAGE_NAME}",
        "--cov-report=term-missing",
        "tests",
    )


#
# Packaging
#


@nox.session(python=PYTHON_VERSION)
def build(session):
    """
    Equivalent of:
      make build
    """
    uv(session, "sync", "--group", "dev")
    session.run("python", "-m", "build")
