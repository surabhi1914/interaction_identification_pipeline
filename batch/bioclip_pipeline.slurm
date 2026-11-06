#!/bin/bash
#SBATCH --job-name=pipeline_bioclip
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --gpus=1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=24:00:00
#SBATCH --output=logs/bioclip2_%j.out
#SBATCH --error=logs/bioclip2_%j.err

set -e

PROJECT_DIR="/rs1/researchers/a/amallav/species_classification/ecological_informatics"

PYTHON="/usr/local/usrapps/ftrscape/snair3/env_AiPipeline/bin/python"

SCRIPT="$PROJECT_DIR/notebooks/interaction/bioclip_full_run.py"

cd "$PROJECT_DIR"

echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Started: $(date)"

nvidia-smi

"$PYTHON" -u "$SCRIPT"

echo "Completed: $(date)"