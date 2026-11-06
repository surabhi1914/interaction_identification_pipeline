#!/bin/bash
#SBATCH --job-name=gdino_24k
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --gpus=1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=24:00:00
#SBATCH --output=/rs1/researchers/a/amallav/species_classification/ecological_informatics/logs/gdino_%j.out
#SBATCH --error=/rs1/researchers/a/amallav/species_classification/ecological_informatics/logs/gdino_%j.err


set -euo pipefail

PROJECT_DIR="/rs1/researchers/a/amallav/species_classification/ecological_informatics"
OUTPUT_DIR="${PROJECT_DIR}/notebooks/object_detection/output/batch_runs"
PY="/usr/local/usrapps/ftrscape/$USER/env_AiPipeline/bin/python"

echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: $(hostname)"
echo "Python: ${PY}"
echo "Start time: $(date)"

nvidia-smi

for BATCH_ID in 0 1 2; do
    echo "Starting batch ${BATCH_ID} at $(date)"

    "$PY" -u "${PROJECT_DIR}/notebooks/object_detection/run_grounding_dino.py" \
        --project-dir "${PROJECT_DIR}" \
        --batch-id "$BATCH_ID" \
        --batch-size 8000 \
        --save-every 100 \
        --output-dir "$OUTPUT_DIR"

    echo "Finished batch ${BATCH_ID} at $(date)"
done

echo "All three batches finished at $(date)"
