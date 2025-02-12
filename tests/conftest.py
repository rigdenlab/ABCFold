import json
import logging
from collections import namedtuple
from pathlib import Path

import pytest

from abcfold.processoutput.alphafold3 import AlphafoldOutput
from abcfold.processoutput.boltz import BoltzOutput
from abcfold.processoutput.chai import ChaiOutput

logger = logging.getLogger("logger")


# Code taken from SliceNDice
@pytest.fixture(scope="session")
def test_data():
    """
    Return a namedtuple object with the paths to all the data files we require.

    Args:
        None

    Returns:
        namedtuple: A namedtuple object with the paths to all the data files we require.
    """
    data_dir = Path("./test_data")
    if not data_dir.exists():
        data_dir = Path("tests/test_data")
    if not data_dir.exists():
        msg = "Could not find the test_data, Please make sure that you're running the \
tests from the root of the repository or the tests directory"
        logger.critical(msg)
        raise FileNotFoundError()
    d = {}

    for test_file in data_dir.glob("*"):

        stem, suffix = test_file.stem, test_file.suffix[1:]
        d[f"test_{stem.replace('-', '_')}_{suffix}"] = str(test_file)

    nt = namedtuple("TestData", d)
    n = nt(**d)

    yield n


@pytest.fixture(scope="session")
def output_objs():
    data_dir = Path("./test_data")
    if not data_dir.exists():
        data_dir = Path("tests/test_data")
    if not data_dir.exists():
        msg = "Could not find the test_data, Please make sure that you're running the \
    tests from the root of the repository or the tests directory"
        logger.critical(msg)
        raise FileNotFoundError()
    d = {}

    adir = data_dir.joinpath("alphafold3_6BJ9")
    bdir = data_dir.joinpath("boltz-1_6BJ9")
    cdir = data_dir.joinpath("chai1_6BJ9")
    name = "6BJ9"
    input_params = adir.joinpath("6bj9_data.json")

    with open(input_params, "r") as f:
        input_params = json.load(f)

    af3_output = AlphafoldOutput(
        adir,
        input_params.copy(),
        name,
    )
    boltz_output = BoltzOutput(
        bdir,
        input_params.copy(),
        name,
    )

    chai_output = ChaiOutput(
        cdir,
        input_params.copy(),
        name,
    )

    d["af3_output"] = af3_output
    d["boltz_output"] = boltz_output
    d["chai_output"] = chai_output

    nt = namedtuple("output_objs", d)
    n = nt(**d)

    yield n
