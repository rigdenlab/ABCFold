from pathlib import Path


def test_process_af3_output(test_data, output_objs):
    af3_output = output_objs.af3_output
    assert af3_output.output_dir.relative_to(af3_output.output_dir.parent) == Path(
        test_data.test_alphafold3_6BJ9_
    ).relative_to("tests/test_data")

    assert "seed-1" in af3_output.output

    files = af3_output.output["seed-1"]

    assert 0 in files
    assert 1 in files

    assert "cif" in files[0]
    assert "af3_pae" in files[0]

    assert "cif" in files[1]
    assert "af3_pae" in files[1]

    json_file = af3_output.af3_pae_files["seed-1"][0]

    assert "atom_plddts" in json_file.data
    assert "pae" in json_file.data
