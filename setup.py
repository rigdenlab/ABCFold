from distutils.command.build import build  # type: ignore

from setuptools import setup


class BuildCommand(build):
    user_options = build.user_options + [
        (
            "script-python-path=",
            None,
            "Path to Python interpreter to be included in the scripts",
        )
    ]

    def initialize_options(self):
        build.initialize_options(self)
        self.script_python_path = None

    def finalize_options(self):
        build.finalize_options(self)

    def run(self):
        global script_python_path
        script_python_path = self.script_python_path
        build.run(self)


with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    cmdclass={"build": BuildCommand},
    long_description=long_description,
    long_description_content_type="text/markdown",
)
