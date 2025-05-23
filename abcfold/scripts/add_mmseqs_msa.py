#!/usr/bin/env python

import gzip
import json
import logging
import math
import os
import random
import shutil
import subprocess
import tarfile
import tempfile
import time
from io import StringIO
from pathlib import Path
from typing import List, Sequence, Union

import pandas as pd
import requests  # type: ignore
from tqdm.autonotebook import tqdm

from abcfold.argparse_utils import (custom_template_argpase_util,
                                    mmseqs2_argparse_util)
from abcfold.scripts.abc_script_utils import (align_and_map,
                                              extract_sequence_from_mmcif,
                                              get_custom_template, get_mmcif)

logger = logging.getLogger("logger")

TQDM_BAR_FORMAT = (
    "{l_bar}{bar}| {n_fmt}/{total_fmt} [elapsed: {elapsed} remaining: {remaining}]"
)

MODULE_OUTPUT_POS = {
    "align":        4,
    "convertalis":  4,
    "expandaln":    5,
    "filterresult": 4,
    "lndb":         2,
    "mergedbs":     2,
    "mvdb":         2,
    "pairaln":      4,
    "result2msa":   4,
    "search":       3,
}


class MMseqs2Exception(Exception):
    def __init__(self):

        msg = "MMseqs2 API is giving errors. Please confirm your input is a valid \
protein sequence. If error persists, please try again an hour later."
        logger.error(msg)
        super().__init__()


def add_msa_to_json(
    input_json,
    mmseqs_db,
    templates,
    num_templates,
    chai_template_output,
    custom_template,
    custom_template_chain,
    target_id,
    input_params=None,
    output_json=None,
    to_file=True,
):
    if input_params is None:
        with open(input_json, "r") as f:
            input_params = json.load(f)

    for sequence in input_params["sequences"]:
        if "protein" in sequence:
            input_id = sequence["protein"]["id"]
            input_sequence = sequence["protein"]["sequence"]
            with tempfile.TemporaryDirectory() as tmpdir:
                if mmseqs_db:
                    logger.info(f"Running Local MMseqs2 on sequence: {input_sequence}")
                    if templates:
                        a3m_lines, templates = run_local_mmseqs(
                            input_sequence,
                            Path(tmpdir),
                            use_templates=True,
                            num_templates=num_templates,
                            mmseqs_db=Path(mmseqs_db),
                        )
                    else:
                        a3m_lines = run_local_mmseqs(
                            input_sequence,
                            Path(tmpdir),
                            use_templates=False,
                            mmseqs_db=Path(mmseqs_db),
                        )
                else:
                    logger.info(f"Running MMseqs2 on sequence: {input_sequence}")
                    # Run MMseqs2 to get unpaired MSA
                    if templates:
                        a3m_lines, templates = run_mmseqs(
                            input_sequence,
                            tmpdir,
                            use_templates=True,
                            num_templates=num_templates,
                        )

                        for i in input_id:
                            table = pd.read_csv(
                                f"{tmpdir}/pdb70.m8",
                                delimiter="\t",
                                header=None,
                                names=[
                                    "query_id",
                                    "subject_id",
                                    "pident",
                                    "length",
                                    "mismatch",
                                    "gapopen",
                                    "query_start",
                                    "query_end",
                                    "subject_start",
                                    "subject_end",
                                    "evalue",
                                    "bitscore",
                                    "comment",
                                ],
                            )

                            table["query_id"] = i

                            if chai_template_output:
                                if os.path.exists(chai_template_output):
                                    table.to_csv(
                                        chai_template_output,
                                        sep="\t",
                                        index=False,
                                        header=False,
                                        mode="a",
                                    )
                                else:
                                    table.to_csv(
                                        chai_template_output,
                                        sep="\t",
                                        index=False,
                                        header=False,
                                    )

                    else:
                        a3m_lines = run_mmseqs(
                            input_sequence,
                            tmpdir,
                            use_templates=False)
                        templates = []

                if custom_template:
                    for template in custom_template:
                        if not os.path.exists(template):
                            msg = f"Custom template file {template} not found"
                            logger.critical(msg)
                            raise FileNotFoundError()
                        # Can only add templates to protein sequences, so check if there
                        # are multiple protein sequences in the input json
                        if (
                            len(
                                [
                                    x
                                    for x in input_params["sequences"]
                                    if "protein" in x.keys()
                                ]
                            )
                            > 1
                            and not target_id
                        ):
                            msg = "Multiple sequences found in input json. \
Please specify target id so that custom template can be added to the correct sequence"
                            raise ValueError(msg)

                    if target_id and len(target_id) > 1:
                        if (len(custom_template) != len(target_id)) or (
                            len(custom_template_chain) != len(target_id)
                        ):
                            msg = "If providing templates for multiple targets, the \
number of target ids must match the number of custom templates and custom template \
chains"
                            raise ValueError(msg)
                        custom_templates = zip(
                            target_id, custom_template, custom_template_chain
                        )
                    else:
                        if len(custom_template) != len(custom_template_chain):
                            msg = "Number of custom templates must match the number of \
custom template chains"
                            raise ValueError(msg)
                        # if a single target id is provided, assume all custom templates
                        # are for the same target
                        if target_id:
                            target_ids = [target_id[0]] * len(custom_template)
                        else:
                            target_ids = [None] * len(custom_template)
                        custom_templates = zip(
                            target_ids, custom_template, custom_template_chain
                        )

                    for i in custom_templates:
                        tid, c_tem, c_tem_chn = i
                        sequence = get_custom_template(
                            sequence,
                            tid,
                            c_tem,
                            c_tem_chn,
                        )

                # Add unpaired MSA to the json
                sequence["protein"]["unpairedMsa"] = a3m_lines[0]
                sequence["protein"]["pairedMsa"] = ""
                sequence["protein"]["templates"] = templates

    if to_file:
        if output_json:
            with open(output_json, "w") as f:
                json.dump(input_params, f)
        else:
            output_json = input_json.replace(".json", "_mmseqs.json")
            with open(output_json, "w") as f:
                json.dump(input_params, f)

    return input_params


# Lightly modified code from https://github.com/sokrypton/ColabFold
def run_mmseqs(
    x,
    prefix,
    use_env=True,
    use_filter=True,
    use_templates=False,
    filter=None,
    use_pairing=False,
    host_url="https://a3m.mmseqs.com",
    num_templates=20
) -> Sequence[object]:
    submission_endpoint = "ticket/pair" if use_pairing else "ticket/msa"

    def submit(seqs, mode, N=101):
        n, query = N, ""
        for seq in seqs:
            query += f">{n}\n{seq}\n"
            n += 1

        res = requests.post(
            f"{host_url}/{submission_endpoint}", data={"q": query, "mode": mode}
        )
        try:
            out = res.json()
        except ValueError:
            logger.error(f"Server didn't reply with json: {res.text}")
            out = {"status": "ERROR"}
        return out

    def status(ID):
        res = requests.get(f"{host_url}/ticket/{ID}")
        try:
            out = res.json()
        except ValueError:
            logger.error(f"Server didn't reply with json: {res.text}")
            out = {"status": "ERROR"}
        return out

    def download(ID, path):
        res = requests.get(f"{host_url}/result/download/{ID}")
        with open(path, "wb") as out:
            out.write(res.content)

    # process input x
    seqs = [x] if isinstance(x, str) else x

    # compatibility to old option
    if filter is not None:
        use_filter = filter

    # setup mode
    if use_filter:
        mode = "env" if use_env else "all"
    else:
        mode = "env-nofilter" if use_env else "nofilter"

    if use_pairing:
        mode = ""
        use_templates = False
        use_env = False

    # define path
    path = prefix
    if not os.path.isdir(path):
        os.mkdir(path)

    # call mmseqs2 api
    tar_gz_file = f"{path}/out.tar.gz"
    N, REDO = 101, True

    # deduplicate and keep track of order
    seqs_unique = list(set(seqs))
    Ms = [N + seqs_unique.index(seq) for seq in seqs]
    # lets do it!
    if not os.path.isfile(tar_gz_file):
        TIME_ESTIMATE = 150 * len(seqs_unique)
        with tqdm(total=TIME_ESTIMATE, bar_format=TQDM_BAR_FORMAT) as pbar:
            while REDO:
                pbar.set_description("SUBMIT")

                # Resubmit job until it goes through
                out = submit(seqs_unique, mode, N)
                while out["status"] in ["UNKNOWN", "RATELIMIT"]:
                    sleep_time = 5 + random.randint(0, 5)
                    logger.info(f"Sleeping for {sleep_time}s. Reason: {out['status']}")
                    # resubmit
                    time.sleep(sleep_time)
                    out = submit(seqs_unique, mode, N)

                if out["status"] == "ERROR":
                    raise MMseqs2Exception()

                if out["status"] == "MAINTENANCE":
                    raise MMseqs2Exception()

                # wait for job to finish
                ID, TIME = out["id"], 0
                pbar.set_description(out["status"])
                while out["status"] in ["UNKNOWN", "RUNNING", "PENDING"]:
                    t = 5 + random.randint(0, 5)
                    logger.info(f"Sleeping for {t}s. Reason: {out['status']}")
                    time.sleep(t)
                    out = status(ID)
                    pbar.set_description(out["status"])
                    if out["status"] == "RUNNING":
                        TIME += t
                        pbar.update(n=t)

                if out["status"] == "COMPLETE":
                    if TIME < TIME_ESTIMATE:
                        pbar.update(n=(TIME_ESTIMATE - TIME))
                    REDO = False

                if out["status"] == "ERROR":
                    REDO = False
                    raise MMseqs2Exception()

            # Download results
            download(ID, tar_gz_file)

    # prep list of a3m files
    if use_pairing:
        a3m_files = [f"{path}/pair.a3m"]
    else:
        a3m_files = [f"{path}/uniref.a3m"]
        if use_env:
            a3m_files.append(f"{path}/bfd.mgnify30.metaeuk30.smag30.a3m")

    # extract a3m files
    if any(not os.path.isfile(a3m_file) for a3m_file in a3m_files):
        with tarfile.open(tar_gz_file) as tar_gz:
            tar_gz.extractall(path)

    # gather a3m lines
    a3m_lines: dict = {}
    for a3m_file in a3m_files:
        a3m_lines = get_a3m_lines(a3m_file)
    a3m_lines_list = ["".join(a3m_lines[n]) for n in Ms]

    if use_templates:
        templates = get_templates(
                x,
                Path(prefix),
                "pdb70.m8",
                num_templates,
            )

    return (a3m_lines_list, templates) if use_templates else a3m_lines_list


def run_mmseqs_command(mmseqs: Path, params: List[Union[str, Path]]):
    module = str(params[0])
    if module in MODULE_OUTPUT_POS:
        output_pos = MODULE_OUTPUT_POS[module]
        output_path = Path(params[output_pos]).with_suffix('.dbtype')
        if output_path.exists():
            logger.info(f"Skipping {module} because {output_path} already exists")
            return

    params_log = " ".join(str(i) for i in params)
    logger.info(f"Running {mmseqs} {params_log}")
    # hide MMseqs2 verbose paramters list that clogs up the log
    os.environ["MMSEQS_CALL_DEPTH"] = "1"
    subprocess.check_call([str(mmseqs)] + [str(i) for i in params])


# Lightly modified code from https://github.com/sokrypton/ColabFold
def run_local_mmseqs(
    x,
    base,
    use_env=True,
    use_templates=False,
    filter=0,
    num_templates=20,
    mmseqs_db=None,
    expand_eval: float = math.inf,
    align_eval: int = 10,
    diff: int = 3000,
    qsc: float = -20.0,
    max_accept: int = 1000000,
    prefilter_mode: int = 0,
    s: float = 8,
    db_load_mode: int = 2,
    threads: int = 32,
    gpu: int = 0,
    gpu_server: int = 0,
    unpack: bool = True,
) -> Sequence[object]:

    if filter:
        # 0.1 was not used in benchmarks due to POSIX shell bug in line above
        #  EXPAND_EVAL=0.1
        align_eval = 10
        qsc = 0.8
        max_accept = 100000

    mmseqs = Path("mmseqs")
    uniref_db = Path("uniref30_2302_db")
    metagenomic_db = Path("colabfold_envdb_202108_db")
    template_db = Path("pdb100_230517")

    base.mkdir(exist_ok=True, parents=True)
    query_file = base.joinpath("query.fas")
    with query_file.open("w") as f:
        query_seq_headername = 101
        f.write(f">{query_seq_headername}\n{x}\n")

    run_mmseqs_command(
        mmseqs,
        ["createdb", query_file, base.joinpath("qdb"), "--shuffle", "0"],
    )

    used_dbs = [uniref_db]
    if use_templates:
        used_dbs.append(template_db)
    if use_env:
        used_dbs.append(metagenomic_db)

    for db in used_dbs:
        if not mmseqs_db.joinpath(f"{db}.dbtype").is_file():
            raise FileNotFoundError(f"Database {db} does not exist")
        if (
            (
                not mmseqs_db.joinpath(f"{db}.idx").is_file()
                and not mmseqs_db.joinpath(f"{db}.idx.index").is_file()
            )
            or os.environ.get("MMSEQS_IGNORE_INDEX", False)
        ):
            logger.info("Search does not use index")
            db_load_mode = 0
            dbSuffix1 = "_seq"
            dbSuffix2 = "_aln"
            dbSuffix3 = ""
        else:
            dbSuffix1 = ".idx"
            dbSuffix2 = ".idx"
            dbSuffix3 = ".idx"

    search_param = ["--num-iterations", "3",
                    "--db-load-mode", str(db_load_mode),
                    "-a", "-e", "0.1", "--max-seqs", "10000"]
    if gpu:
        # gpu version only supports ungapped prefilter currently
        search_param += ["--gpu", str(gpu), "--prefilter-mode", "1"]
    else:
        search_param += ["--prefilter-mode", str(prefilter_mode)]
        # sensitivy can only be set for non-gpu version,
        # gpu version runs at max sensitivity
        if s is not None:
            search_param += ["-s", "{:.1f}".format(s)]
        else:
            search_param += ["--k-score", "'seq:96,prof:80'"]
    if gpu_server:
        search_param += ["--gpu-server", str(gpu_server)]

    filter_param = ["--filter-msa", str(filter),
                    "--filter-min-enable", "1000",
                    "--diff", str(diff),
                    "--qid", "0.0,0.2,0.4,0.6,0.8,1.0",
                    "--qsc", "0", "--max-seq-id", "0.95"]
    expand_param = ["--expansion-mode", "0",
                    "-e", str(expand_eval),
                    "--expand-filter-clusters", str(filter),
                    "--max-seq-id", "0.95"]

    if not base.joinpath("uniref.a3m").with_suffix('.a3m.dbtype').exists():
        run_mmseqs_command(mmseqs,
                           ["search", base.joinpath("qdb"),
                            mmseqs_db.joinpath(uniref_db),
                            base.joinpath("res"),
                            base.joinpath("tmp"),
                            "--threads", str(threads)] + search_param)
        run_mmseqs_command(mmseqs,
                           ["mvdb",
                            base.joinpath("tmp/latest/profile_1"),
                            base.joinpath("prof_res")])
        run_mmseqs_command(mmseqs,
                           ["lndb",
                            base.joinpath("qdb_h"),
                            base.joinpath("prof_res_h")])
        run_mmseqs_command(mmseqs,
                           ["expandaln",
                            base.joinpath("qdb"),
                            mmseqs_db.joinpath(f"{uniref_db}{dbSuffix1}"),
                            base.joinpath("res"),
                            mmseqs_db.joinpath(f"{uniref_db}{dbSuffix2}"),
                            base.joinpath("res_exp"),
                            "--db-load-mode", str(db_load_mode),
                            "--threads", str(threads)] + expand_param)
        run_mmseqs_command(mmseqs,
                           ["align",
                            base.joinpath("prof_res"),
                            mmseqs_db.joinpath(f"{uniref_db}{dbSuffix1}"),
                            base.joinpath("res_exp"),
                            base.joinpath("res_exp_realign"),
                            "--db-load-mode", str(db_load_mode),
                            "-e", str(align_eval),
                            "--max-accept", str(max_accept),
                            "--threads", str(threads),
                            "--alt-ali", "10", "-a"])
        run_mmseqs_command(mmseqs,
                           ["filterresult",
                            base.joinpath("qdb"),
                            mmseqs_db.joinpath(f"{uniref_db}{dbSuffix1}"),
                            base.joinpath("res_exp_realign"),
                            base.joinpath("res_exp_realign_filter"),
                            "--db-load-mode",
                            str(db_load_mode),
                            "--qid", "0",
                            "--qsc", str(qsc),
                            "--diff", "0",
                            "--threads", str(threads),
                            "--max-seq-id", "1.0",
                            "--filter-min-enable", "100"])
        run_mmseqs_command(mmseqs,
                           ["result2msa",
                            base.joinpath("qdb"),
                            mmseqs_db.joinpath(f"{uniref_db}{dbSuffix1}"),
                            base.joinpath("res_exp_realign_filter"),
                            base.joinpath("uniref.a3m"),
                            "--msa-format-mode", "6",
                            "--db-load-mode", str(db_load_mode),
                            "--threads", str(threads)] + filter_param)
        run_mmseqs_command(mmseqs, ["rmdb", base.joinpath("res_exp_realign_filter")])
        run_mmseqs_command(mmseqs, ["rmdb", base.joinpath("res_exp_realign")])
        run_mmseqs_command(mmseqs, ["rmdb", base.joinpath("res_exp")])
        run_mmseqs_command(mmseqs, ["rmdb", base.joinpath("res")])
    else:
        logger.info(f"Skipping {uniref_db} search because uniref.a3m already exists")

    bfd_exists = base.joinpath(
        "bfd.mgnify30.metaeuk30.smag30.a3m"
    ).with_suffix('.a3m.dbtype').exists()
    if use_env and not bfd_exists:
        run_mmseqs_command(mmseqs,
                           ["search",
                            base.joinpath("prof_res"),
                            mmseqs_db.joinpath(metagenomic_db),
                            base.joinpath("res_env"),
                            base.joinpath("tmp3"),
                            "--threads", str(threads)] + search_param)
        run_mmseqs_command(mmseqs,
                           ["expandaln",
                            base.joinpath("prof_res"),
                            mmseqs_db.joinpath(f"{metagenomic_db}{dbSuffix1}"),
                            base.joinpath("res_env"),
                            mmseqs_db.joinpath(f"{metagenomic_db}{dbSuffix2}"),
                            base.joinpath("res_env_exp"), "-e", str(expand_eval),
                            "--expansion-mode", "0",
                            "--db-load-mode", str(db_load_mode),
                            "--threads", str(threads)])
        run_mmseqs_command(mmseqs,
                           ["align",
                            base.joinpath("tmp3/latest/profile_1"),
                            mmseqs_db.joinpath(f"{metagenomic_db}{dbSuffix1}"),
                            base.joinpath("res_env_exp"),
                            base.joinpath("res_env_exp_realign"),
                            "--db-load-mode", str(db_load_mode),
                            "-e", str(align_eval),
                            "--max-accept", str(max_accept),
                            "--threads", str(threads),
                            "--alt-ali", "10", "-a"])
        run_mmseqs_command(mmseqs,
                           ["filterresult",
                            base.joinpath("qdb"),
                            mmseqs_db.joinpath(f"{metagenomic_db}{dbSuffix1}"),
                            base.joinpath("res_env_exp_realign"),
                            base.joinpath("res_env_exp_realign_filter"),
                            "--db-load-mode",
                            str(db_load_mode),
                            "--qid", "0",
                            "--qsc", str(qsc),
                            "--diff", "0",
                            "--max-seq-id", "1.0",
                            "--threads", str(threads),
                            "--filter-min-enable", "100"])
        run_mmseqs_command(mmseqs,
                           ["result2msa",
                            base.joinpath("qdb"),
                            mmseqs_db.joinpath(f"{metagenomic_db}{dbSuffix1}"),
                            base.joinpath("res_env_exp_realign_filter"),
                            base.joinpath("bfd.mgnify30.metaeuk30.smag30.a3m"),
                            "--msa-format-mode", "6",
                            "--db-load-mode", str(db_load_mode),
                            "--threads", str(threads)] + filter_param)
        run_mmseqs_command(mmseqs,
                           ["rmdb", base.joinpath("res_env_exp_realign_filter")])
        run_mmseqs_command(mmseqs, ["rmdb", base.joinpath("res_env_exp_realign")])
        run_mmseqs_command(mmseqs, ["rmdb", base.joinpath("res_env_exp")])
        run_mmseqs_command(mmseqs, ["rmdb", base.joinpath("res_env")])
    elif use_env:
        logger.info(
            f"Skipping {metagenomic_db} search because \
bfd.mgnify30.metaeuk30.smag30.a3m already exists")

    tmpl_db_exists = base.joinpath(
        f"{template_db}.m8"
    ).with_suffix('.m8.dbtype').exists()
    if use_templates and not tmpl_db_exists:
        run_mmseqs_command(mmseqs,
                           ["search",
                            base.joinpath("prof_res"),
                            mmseqs_db.joinpath(template_db),
                            base.joinpath("res_pdb"),
                            base.joinpath("tmp2"),
                            "--db-load-mode", str(db_load_mode),
                            "--threads", str(threads),
                            "-s", "7.5",
                            "-a", "-e", "0.1",
                            "--prefilter-mode", str(prefilter_mode)])
        run_mmseqs_command(mmseqs,
                           ["convertalis",
                            base.joinpath("prof_res"),
                            mmseqs_db.joinpath(f"{template_db}{dbSuffix3}"),
                            base.joinpath("res_pdb"),
                            base.joinpath(f"{template_db}"),
                            "--format-output",
                            "query,target,fident,alnlen,mismatch,\
gapopen,qstart,qend,tstart,tend,evalue,bits,cigar",
                            "--db-output", "1",
                            "--db-load-mode", str(db_load_mode),
                            "--threads", str(threads)])
        run_mmseqs_command(mmseqs, ["rmdb", base.joinpath("res_pdb")])
    elif use_templates:
        logger.info(
            f"Skipping {template_db} search because {template_db}.m8 already exists"
            )

    if use_env:
        run_mmseqs_command(mmseqs,
                           ["mergedbs", base.joinpath("qdb"),
                            base.joinpath("final.a3m"),
                            base.joinpath("uniref.a3m"),
                            base.joinpath("bfd.mgnify30.metaeuk30.smag30.a3m")])
        run_mmseqs_command(mmseqs,
                           ["rmdb",
                            base.joinpath("bfd.mgnify30.metaeuk30.smag30.a3m")])
        run_mmseqs_command(mmseqs,
                           ["rmdb",
                            base.joinpath("uniref.a3m")])
    else:
        run_mmseqs_command(mmseqs,
                           ["mvdb",
                            base.joinpath("uniref.a3m"),
                            base.joinpath("final.a3m")])
        run_mmseqs_command(mmseqs,
                           ["rmdb",
                            base.joinpath("uniref.a3m")])

    if unpack:
        run_mmseqs_command(mmseqs,
                           ["unpackdb",
                            base.joinpath("final.a3m"),
                            base.joinpath("."),
                            "--unpack-name-mode", "0",
                            "--unpack-suffix", ".a3m"])
        run_mmseqs_command(mmseqs,
                           ["rmdb",
                            base.joinpath("final.a3m")])

        if use_templates:
            run_mmseqs_command(mmseqs,
                               ["unpackdb",
                                base.joinpath(f"{template_db}"),
                                base.joinpath("."),
                                "--unpack-name-mode", "0",
                                "--unpack-suffix", ".m8"])
            if base.joinpath(f"{template_db}").exists():
                run_mmseqs_command(mmseqs, ["rmdb", base.joinpath(f"{template_db}")])

    run_mmseqs_command(mmseqs, ["rmdb", base.joinpath("prof_res")])
    run_mmseqs_command(mmseqs, ["rmdb", base.joinpath("prof_res_h")])
    shutil.rmtree(base.joinpath("tmp"))
    if use_templates:
        shutil.rmtree(base.joinpath("tmp2"))
    if use_env:
        shutil.rmtree(base.joinpath("tmp3"))

    if unpack:
        run_mmseqs_command(mmseqs, ["rmdb", base.joinpath("qdb")])
        run_mmseqs_command(mmseqs, ["rmdb", base.joinpath("qdb_h")])
        output_a3m = base.joinpath("0.a3m")
    else:
        output_a3m = base.joinpath("final.a3m")

    query_file.unlink()

    # Gather a3m lines in the same way as the API
    seqs = [x] if isinstance(x, str) else x
    N = 101
    seqs_unique = list(set(seqs))
    Ms = [N + seqs_unique.index(seq) for seq in seqs]
    a3m_lines = get_a3m_lines(output_a3m)
    a3m_lines_list = ["".join(a3m_lines[n]) for n in Ms]

    if use_templates:
        templates = get_templates(
            x,
            base,
            "0.m8",
            num_templates,
            mmseqs_db=mmseqs_db,
        )

    return (a3m_lines_list, templates) if use_templates else a3m_lines_list


def get_a3m_lines(output_a3m):
    a3m_lines: dict = {}
    update_M, M = True, None
    for line in open(output_a3m, "r"):
        if len(line) > 0:
            if "\x00" in line:
                line = line.replace("\x00", "")
                update_M = True
            if line.startswith(">") and update_M:
                M = int(line[1:].rstrip())
                update_M = False
                if M not in a3m_lines:
                    a3m_lines[M] = []
            a3m_lines[M].append(line)

    return a3m_lines


def get_templates(x, base, m8, num_templates, mmseqs_db=None):
    tested_pdbs = []
    templates = []
    logger.info("Finding and preparing templates")
    count = 0
    for line in open(base.joinpath(m8), "r"):
        template = {}
        if count < num_templates:
            p = line.rstrip().split()
            pdb, qid, alilen, tstart, tend = (
                p[1],
                float(p[2]),
                float(p[3]),
                int(p[8]),
                int(p[9]),
            )
            coverage = alilen / len(x)
            pdb_id = pdb.split("_")[0]

            # Use the same template filters as AF3 and only use 1 template per PDB
            if (
                qid == 1.0
                and coverage >= 0.95
                or coverage < 0.1
                or pdb_id in tested_pdbs
            ):
                continue

            pdb_id = pdb.split("_")[0]
            if mmseqs_db:
                cif_str = fetch_local_mmcif(pdb_id,
                                            pdb.split("_")[1],
                                            tstart,
                                            tend,
                                            base,
                                            mmseqs_db)
            else:
                cif_str = fetch_mmcif(pdb_id,
                                      pdb.split("_")[1],
                                      tstart,
                                      tend,
                                      base)
            template["mmcif"] = cif_str

            template_seq = extract_sequence_from_mmcif(StringIO(cif_str))
            query_indices, template_indices = align_and_map(x, template_seq)

            template["queryIndices"] = query_indices
            template["templateIndices"] = template_indices
            templates.append(template)
            tested_pdbs.append(pdb_id)
            count += 1
    logger.info(f"Found the following templates: {tested_pdbs}")
    return templates


def fetch_mmcif(
    pdb_id,
    chain_id,
    start,
    end,
    tmpdir,
):
    """
    Fetch the mmcif file for a given PDB ID
    and chain ID and prepare it for use in AlphaFold3
    """
    pdb_id = pdb_id.lower()
    url_base = "http://www.ebi.ac.uk/pdbe-srv/view/files/"
    url = url_base + pdb_id + ".cif"
    response = requests.get(url)
    text = response.text

    output = os.path.join(tmpdir, pdb_id + ".cif")
    with open(output, "w") as f:
        f.write(text)

    return get_mmcif(output, pdb_id, chain_id, start, end, tmpdir)


def fetch_local_mmcif(
        pdb_id,
        chain_id,
        start,
        end,
        tmpdir,
        mmseqs_db,
):
    """
    Fetch the mmcif file for a given PDB ID
    and chain ID and prepare it for use in AlphaFold3
    """
    pdb_id = pdb_id.lower()
    assert len(pdb_id) == 4, f"Invalid PDB ID: {pdb_id}"
    inner_code = pdb_id[1:3]
    mmcif_location = mmseqs_db.joinpath(f"pdb/divided/{inner_code}/{pdb_id}.cif.gz")
    if not mmcif_location.exists():
        raise FileNotFoundError(f"MMseqs2 database {mmcif_location} does not exist")
    with gzip.open(mmcif_location, "rb") as f:
        cif_str = f.read().decode("utf-8")

    output = os.path.join(tmpdir, pdb_id + ".cif")
    with open(output, "w") as f:
        f.write(cif_str)

    return get_mmcif(output, pdb_id, chain_id, start, end, tmpdir)


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Add MMseqs2 unpaired MSA to AlphaFold3 json"
    )
    parser.add_argument("--input_json", help="Input alphafold3 json file")
    parser.add_argument("--output_json", help="Output alphafold3 json file")

    parser = mmseqs2_argparse_util(parser)
    parser = custom_template_argpase_util(parser)

    args = parser.parse_args()

    add_msa_to_json(  # pragma: no cover
        args.input_json,
        args.mmseqs_database,
        args.templates,
        args.num_templates,
        False,
        args.custom_template,
        args.custom_template_chain,
        args.target_id,
        output_json=args.output_json,
        to_file=True,
    )


if __name__ == "__main__":  # pragma: no cover
    main()
