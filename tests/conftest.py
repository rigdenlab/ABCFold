import configparser
import json
import logging
import shutil
import tempfile
from collections import namedtuple
from pathlib import Path

import pytest

from abcfold.output.alphafold3 import AlphafoldOutput
from abcfold.output.boltz import BoltzOutput
from abcfold.output.chai import ChaiOutput
from abcfold.output.openfold3 import OpenfoldOutput
from abcfold.output.protenix import ProtenixOutput

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

    config_file = Path("../abcfold/data/config.ini")
    config = configparser.SafeConfigParser()
    config_dict = {}
    config.read(str(config_file))
    for section in config.sections():
        config_dict.update(dict(config.items(section)))
    d["config_dict"] = config_dict

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

    config_file = Path("../abcfold/data/config.ini")
    config = configparser.SafeConfigParser()
    config_dict = {}
    config.read(str(config_file))
    for section in config.sections():
        config_dict.update(dict(config.items(section)))

    adir = data_dir.joinpath("alphafold3_6BJ9")
    bdir = data_dir.joinpath("boltz_6BJ9_seed-1")
    cdir = data_dir.joinpath("chai1_6BJ9_seed-1")
    pdir = data_dir.joinpath("protenix_6BJ9_seed-1")
    odir = data_dir.joinpath("openfold_6BJ9_seed-1")
    name = "6BJ9"
    input_params = adir.joinpath("6bj9_data.json")

    # Create temporary directories
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_adir = Path(temp_dir) / "alphafold3_6BJ9"
        temp_bdir = Path(temp_dir) / "boltz_6BJ9_seed-1"
        temp_cdir = Path(temp_dir) / "chai1_6BJ9_seed-1"
        temp_pdir = Path(temp_dir) / "protenix_6BJ9_seed-1"
        temp_odir = Path(temp_dir) / "openfold_6BJ9_seed-1"

        shutil.copytree(adir, temp_adir)
        shutil.copytree(bdir, temp_bdir)
        shutil.copytree(cdir, temp_cdir)
        shutil.copytree(pdir, temp_pdir)
        shutil.copytree(odir, temp_odir)

        with open(input_params, "r") as f:
            input_params = json.load(f)

        af3_output = AlphafoldOutput(
            temp_adir,
            input_params.copy(),
            name,
        )
        boltz_output = BoltzOutput(
            [temp_bdir],
            input_params.copy(),
            name,
        )

        chai_output = ChaiOutput(
            [temp_cdir],
            input_params.copy(),
            name,
            config_dict,
        )

        protenix_output = ProtenixOutput(
            [temp_pdir],
            input_params.copy(),
            name,
        )

        openfold_output = OpenfoldOutput(
            [temp_odir],
            input_params.copy(),
            name,
        )

        d["af3_output"] = af3_output
        d["boltz_output"] = boltz_output
        d["chai_output"] = chai_output
        d["protenix_output"] = protenix_output
        d["openfold_output"] = openfold_output
        d["config_dict"] = config_dict
        nt = namedtuple("output_objs", d)
        n = nt(**d)

        yield n
