#!/bin/bash
#SBATCH --job-name=bioclip2_full
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --gpus=1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=/rs1/researchers/a/amallav/species_classification/ecological_informatics/logs/bioclip2_10222026_%j.out
#SBATCH --error=/rs1/researchers/a/amallav/species_classification/ecological_informatics/logs/bioclip2_10222026_%j.err

set -euo pipefail

PROJECT_DIR="/rs1/researchers/a/amallav/species_classification/ecological_informatics"

PY="/usr/local/usrapps/ftrscape/$USER/env_AiPipeline/bin/python"

export HF_HOME="${PROJECT_DIR}/models/hf_home"
export HF_HUB_CACHE="${HF_HOME}/hub"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

cd "${PROJECT_DIR}/notebooks/species_identification"

echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Start: $(date)"

nvidia-smi

# Execute the entire Jupyter notebook
"$PY" -m jupyter nbconvert \
    --to notebook \
    --execute run_bioclip.ipynb \
    --output run_bioclip_executed.ipynb \
    --ExecutePreprocessor.timeout=-1 \
    --ExecutePreprocessor.kernel_name=python3

echo "Finished: $(date)"