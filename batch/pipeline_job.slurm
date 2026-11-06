#!/bin/bash
#SBATCH --job-name=pipeline_gd
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --gpus=1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=24:00:00
#SBATCH --output=logs/pipeline_gd_10232026_%j.out
#SBATCH --error=logs/pipeline_gd_10232026_%j.err

set -euo pipefail

PROJECT_DIR="/rs1/researchers/a/amallav/species_classification/ecological_informatics"
PY="/usr/local/usrapps/ftrscape/$USER/env_AiPipeline/bin/python"

cd "$PROJECT_DIR/notebooks/interaction"

echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Started: $(date)"

# Verify GPU
nvidia-smi

# Check PyTorch CUDA compatibility
"$PY" -c '
import torch

print("Torch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
    print("CUDA capability:", torch.cuda.get_device_capability(0))
    x = torch.ones(10, device="cuda")
    torch.cuda.synchronize()
    print("CUDA test passed")
else:
    raise RuntimeError("CUDA not available")
'

# Execute notebook
"$PY" -m jupyter nbconvert \
    --to notebook \
    --execute run_pipeline.ipynb \
    --output "run_pipeline_executed_${SLURM_JOB_ID}.ipynb" \
    --ExecutePreprocessor.timeout=-1 \
    --ExecutePreprocessor.kernel_name=env_AiPipeline

echo "Completed: $(date)"