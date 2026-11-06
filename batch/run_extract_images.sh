#!/usr/bin/env bash
#SBATCH --partition=xfer
#SBATCH --job-name=image_download
#SBATCH --time=04:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --output=logs/image_download_%j.out
#SBATCH --error=logs/image_download_%j.err


set -euo pipefail

PROJECT_DIR="/rs1/researchers/a/amallav/species_classification/ecological_informatics"
PY="/usr/local/usrapps/ftrscape/$USER/env_AiPipeline/bin/python"

cd "$PROJECT_DIR"

echo "HOST=$(hostname)"
echo "START=$(date)"
echo "PY=$PY"


"$PY" -u notebooks/image_extraction/fetch_images.py \
    --workers 2 \
    --report-every 1


echo "END=$(date)"