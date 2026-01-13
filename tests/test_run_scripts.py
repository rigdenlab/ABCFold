import subprocess


def test_alphafold3_script():
    with subprocess.Popen(
        "python -m abcfold.abcfold --help",
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    ) as p:
        stdout, stderr = p.communicate()
        print(stdout)
        print(stderr)
        assert p.returncode == 0, f"Return code: {p.returncode}\n{stderr.decode()}"
        assert b"usage: abcfold" in stdout, (
            f"stdout: {stdout.decode()}\nstderr: {stderr.decode()}"
        )


def test_add_custom_template_script():
    with subprocess.Popen(
        "python -m abcfold.scripts.add_custom_template --help",
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    ) as p:
        stdout, stderr = p.communicate()
        assert p.returncode == 0, f"Return code: {p.returncode}\n{stderr.decode()}"
        assert b"usage: add_custom_template" in stdout, (
            f"stdout: {stdout.decode()}\nstderr: {stderr.decode()}"
        )


def test_add_mmseqs_msa_script():
    with subprocess.Popen(
        "python -m abcfold.scripts.add_mmseqs_msa --help",
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    ) as p:
        stdout, stderr = p.communicate()
        assert p.returncode == 0, f"Return code: {p.returncode}\n{stderr.decode()}"
        assert b"usage: add_mmseqs_msa" in stdout, (
            f"stdout: {stdout.decode()}\nstderr: {stderr.decode()}"
        )


def test_ipsae_script():
    with subprocess.Popen(
        "python -m abcfold.scripts.ipsae --help",
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    ) as p:
        stdout, stderr = p.communicate()
        assert p.returncode == 0, f"Return code: {p.returncode}\n{stderr.decode()}"
        assert b"usage: ipsae" in stdout, (
            f"stdout: {stdout.decode()}\nstderr: {stderr.decode()}"
        )
