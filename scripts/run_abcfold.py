#!/usr/bin/env python3
from pathlib import Path
import subprocess, os, queue, threading, time, argparse, logging
import torch
import typer
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TimeElapsedColumn, TextColumn, BarColumn, TaskProgressColumn, TimeRemainingColumn
from rich.table import Table
from rich.logging import RichHandler

app = typer.Typer()
console = Console()

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(rich_tracebacks=True)]
)

input_dir = Path("json_inputs")
output_root = Path("outputs")
log_file = Path("run.log")
if log_file.exists():
    log_file.unlink()

methods = [
    {"name": "af3",   "prefix": "alphafold3", "log": "af3_error.log"},
    {"name": "boltz", "prefix": "boltz",      "log": "boltz_error.log"},
    {"name": "chai",  "prefix": "chai",       "log": "chai_error.log"},
]

def status(jf: Path):
    od = output_root / jf.stem
    if not od.is_dir():
        return {m["name"]: "missing" for m in methods}
    names = {p.name for p in od.iterdir()}
    r = {}
    for m in methods:
        if m["log"] in names:
            r[m["name"]] = "failed"
        elif any(n.startswith(m["prefix"]) and not n.endswith("_error.log") for n in names):
            r[m["name"]] = "success"
        else:
            r[m["name"]] = "missing"
    return r

def print_stats(files, title):
    tot = len(files)
    agg = {m["name"]: {"success": 0, "failed": 0, "missing": 0} for m in methods}
    for f in files:
        s = status(f)
        for k, v in s.items():
            agg[k][v] += 1
    table = Table(title=f"{title} (total {tot})")
    table.add_column("Method", style="cyan")
    table.add_column("Success", style="green")
    table.add_column("Failed", style="red")
    table.add_column("Missing", style="yellow")
    for k, c in agg.items():
        table.add_row(k, str(c["success"]), str(c["failed"]), str(c["missing"]))
    console.print(table)

def run_job(jf: Path, gpu: str):
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = "" if gpu == "cpu" else gpu
    od = output_root / jf.stem
    od.mkdir(parents=True, exist_ok=True)
    cmd = [
        "abcfold",
        str(jf),
        str(od),
        "-abc",
        "--gpus", gpu,
        "--mmseqs2",
        "--override",
        "--number_of_models", "5",
        "--no_visuals",
    ]
    with log_file.open("a") as lg:
        ret = subprocess.call(cmd, stdout=lg, stderr=lg, env=env)
    if ret != 0:
        logging.error(f"Failed to process {jf.name}")
    return ret == 0

def expand(arg):
    if arg.lower() == "cpu":
        return ["cpu"]
    if arg.lower() == "all":
        n = torch.cuda.device_count()
        return [str(i) for i in range(n)] or ["cpu"]
    return [x.strip() for x in arg.split(",") if x.strip()]

def worker(gpu, q: queue.Queue, progress, task):
    while True:
        try:
            jf = q.get_nowait()
        except queue.Empty:
            break
        run_job(jf, gpu)
        progress.update(task, advance=1)
        q.task_done()

@app.command()
def main(
    gpus: str = typer.Option("0", help="all | cpu | 0,1,3 …"),
    input_path: Path = typer.Option(input_dir, help="Input directory with JSON files"),
    output_path: Path = typer.Option(output_root, help="Output directory for results")
):
    global input_dir, output_root
    input_dir = input_path
    output_root = output_path
    gpus = expand(gpus)
    files = sorted(input_dir.glob("*.json"))
    print_stats(files, "BEFORE")
    q = queue.Queue()
    for jf in files:
        if any(status(jf)[m["name"]] != "success" for m in methods):
            q.put(jf)
    total_jobs = q.qsize()
    logging.info(f"Queued {total_jobs} jobs on GPUs {gpus}")
    with Progress(
        SpinnerColumn(),
        TextColumn("Processing files:"),
        BarColumn(bar_width=None),
        TaskProgressColumn("{task.completed}/{task.total}"),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=console,
    ) as progress:
        task = progress.add_task(f"ABCFold Calculation [{total_jobs} jobs]", total=total_jobs)
        threads = [threading.Thread(target=worker, args=(g, q, progress, task), daemon=True) for g in gpus]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
    print_stats(files, "AFTER")

if __name__ == "__main__":
    app()
