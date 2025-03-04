import argparse

from abcfold.argparse_utils import (alphafold_argparse_util,
                                    custom_template_argpase_util,
                                    main_argpase_util, mmseqs2_argparse_util)


def test_mmseqs2_argparse_util():

    parser = argparse.ArgumentParser()
    parser = mmseqs2_argparse_util(parser)
    args = parser.parse_args(["--templates", "--num_templates", "20"])
    assert args.templates
    assert args.num_templates == 20


def test_custom_template_argpase_util():
    parser = argparse.ArgumentParser()
    parser = custom_template_argpase_util(parser)
    args = parser.parse_args(
        [
            "--target_id",
            "A",
            "--custom_template",
            "test_data/test_6BJ9_cif",
            "--custom_template_chain",
            "B",
        ]
    )
    assert args.target_id == ["A"]
    assert args.custom_template == ["test_data/test_6BJ9_cif"]
    assert args.custom_template_chain == ["B"]


def test_custom_template_argpase_util_2():
    parser = argparse.ArgumentParser()
    parser = custom_template_argpase_util(parser)
    args = parser.parse_args(
        [
            "--target_id",
            "A",
            "B",
            "--custom_template",
            "test_data/test_6BJ9_cif",
            "test_data/test_6BJ9_cif",
            "--custom_template_chain",
            "A",
            "B",
        ]
    )
    assert args.target_id == ["A", "B"]
    assert args.custom_template == ["test_data/test_6BJ9_cif",
                                    "test_data/test_6BJ9_cif"]
    assert args.custom_template_chain == ["A", "B"]


def test_af3_argparse_main(test_data):
    parser = argparse.ArgumentParser()
    parser = main_argpase_util(parser)
    parser = alphafold_argparse_util(parser)
    parser = mmseqs2_argparse_util(parser)
    args = parser.parse_args(
        [
            test_data.test_inputA_json,
            "test_data",
            "--output_json",
            "test_data/test_outputA_json",
            "--database",
            "path/to/database",
            "--model_params",
            "path/to/model_params",
            "--mmseqs2",
        ]
    )
    assert args.input_json == test_data.test_inputA_json
    assert args.output_json == "test_data/test_outputA_json"
    assert args.database_dir == "path/to/database"
    assert args.model_params == "path/to/model_params"
    assert args.mmseqs2
