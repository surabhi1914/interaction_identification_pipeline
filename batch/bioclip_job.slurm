#!/bin/bash
#SBATCH --job-name=bioclip2_all
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --gpus=1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=24:00:00
#SBATCH --output=/rs1/researchers/a/amallav/species_classification/ecological_informatics/logs/bioclip2_%j.out
#SBATCH --error=/rs1/researchers/a/amallav/species_classification/ecological_informatics/logs/bioclip2_%j.err
set -euo pipefail

PROJECT_DIR="/rs1/researchers/a/amallav/species_classification/ecological_informatics"

PY="/usr/local/usrapps/ftrscape/$USER/env_AiPipeline/bin/python"

export HF_HOME="${PROJECT_DIR}/models/hf_home"
export HF_HUB_CACHE="${HF_HOME}/hub"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: $(hostname)"
echo "Python: $PY"
echo "HF_HOME: $HF_HOME"
echo "HF_HUB_CACHE: $HF_HUB_CACHE"

# Validate cache before initializing BioCLIP2
"$PY" - <<'PY'
from huggingface_hub import hf_hub_download

for filename in [
    "open_clip_config.json",
    "open_clip_model.safetensors"
]:
    path = hf_hub_download(
        repo_id="imageomics/bioclip-2",
        filename=filename,
        local_files_only=True
    )
    print("Cached:", path)
PY

nvidia-smi


"$PY" -u "${PROJECT_DIR}/notebooks/species_identification/run_bioclip2.py" \
    --project-dir "$PROJECT_DIR" \
    --input "${PROJECT_DIR}/notebooks/object_detection/output/analysis/bioclip2_input_detections.parquet" \
    --output-dir "${PROJECT_DIR}/notebooks/species_identification/output/bioclip2" \
    --top-k 5 \
    --save-every 25