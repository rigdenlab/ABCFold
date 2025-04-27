import os
import subprocess
import shutil
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from loguru import logger

JSON_INPUT_DIR = Path("/mnt/ligandpro/soft/protein/alphafold3/af3_calc_bindingDB/json_inputs")
CPU_OUTPUT_DIR = Path("/mnt/ligandpro/soft/protein/alphafold3/af3_calc_bindingDB/MSA")
GPU_OUTPUT_DIR = Path("/mnt/ligandpro/soft/protein/alphafold3/af3_calc_bindingDB/inference")
MODELS_DIR = Path("/mnt/ligandpro/soft/protein/alphafold3/models")
DB_DIR = Path("/mnt/ligandpro/db/AF3")
IMAGE = "alphafold3"

CPU_LOG = "cpu_stage.log"

logger.remove()
logger.add(CPU_LOG, format="{time:YYYY-MM-DD HH:mm:ss} {message}", level="INFO")

def run_cmd(cmd):
    p = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    if not p.stdout:
        rc = p.wait()
        if rc != 0:
            raise RuntimeError(f"Error {rc}: {cmd}")
        return
    for line in p.stdout:
        logger.info(line.rstrip("\n"))
    rc = p.wait()
    if rc != 0:
        raise RuntimeError(f"Error {rc}: {cmd}")

def cpu_task(json_file: Path):
    base = json_file.stem
    out_dir_name = base.lower()

    cmd = f"""
docker run -i --rm \
--volume {JSON_INPUT_DIR}:/root/af_input \
--volume {GPU_OUTPUT_DIR}:/root/af_output \
--volume {MODELS_DIR}:/root/models \
--volume {DB_DIR}:/root/public_databases \
-e DB_DIR=/root/public_databases \
{IMAGE} \
python run_alphafold.py \
 --json_path=/root/af_input/{json_file.name} \
 --model_dir=/root/models \
 --output_dir=/root/af_input \
 --jackhmmer_n_cpu=8 \
 --nhmmer_n_cpu=8 \
 --run_inference=false
"""
    run_cmd(cmd)

    subdir = JSON_INPUT_DIR / out_dir_name
    data_json = subdir / f"{out_dir_name}_data.json"
    if not data_json.is_file():
        raise FileNotFoundError(f"Enriched JSON not found: {data_json}")

    CPU_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    target_subdir = CPU_OUTPUT_DIR / out_dir_name
    if target_subdir.exists():
        shutil.rmtree(target_subdir)
    shutil.move(str(subdir), str(CPU_OUTPUT_DIR))

def main():
    if not JSON_INPUT_DIR.is_dir():
        logger.error(f"No directory {JSON_INPUT_DIR}")
        return
    json_files = list(JSON_INPUT_DIR.glob("*.json"))
    if not json_files:
        logger.info("No JSON files in input directory.")
        return

    to_process = []
    for jf in json_files:
        out_dir_name = jf.stem.lower()
        data_json = CPU_OUTPUT_DIR / out_dir_name / f"{out_dir_name}_data.json"
        if data_json.is_file():
            logger.info(f"Already processed: {jf.name}, skipping...")
            continue
        to_process.append(jf)

    if not to_process:
        logger.info("No new JSON files to process.")
        return

    with ThreadPoolExecutor(max_workers=4) as pool:
        futs = {pool.submit(cpu_task, jf): jf for jf in to_process}
        for _ in tqdm(as_completed(futs), total=len(futs), desc="CPU stage"):
            pass

if __name__ == "__main__":
    main()
