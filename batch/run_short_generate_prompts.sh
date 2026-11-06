#!/usr/bin/env bash
#BSUB -J grounding_dino_prompts
#BSUB -q short_gpu
#BSUB -n 8
#BSUB -W 119
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -R "select[h200 || h100 || a100 || l40s || l40 || a10]"
#BSUB -o logs/gd_prompts_split1_%J.out
#BSUB -e logs/gd_prompts_split1_%J.err

set -euo pipefail

PROJECT_DIR="/rs1/researchers/a/amallav/species_classification/ecological_informatics"
PY="/usr/local/usrapps/ftrscape/$USER/env_AiPipeline/bin/python"

cd "$PROJECT_DIR"
mkdir -p logs


# Hugging Face and PyTorch caches
export HF_HOME="$PROJECT_DIR/models/cache/_hf_cache"
export TRANSFORMERS_CACHE=$HF_HOME
export HF_HUB_CACHE="$HF_HOME/hub"
export TORCH_HOME="$PROJECT_DIR/models/cache/torch_cache"

mkdir -p "$HF_HUB_CACHE" "$TORCH_HOME"

# project paths
export PROJECT_DIR=$PROJECT_DIR
export DATASET_DIR=$PROJECT_DIR/notebooks/prompt_engineering/data
export MODEL_ID=$PROJECT_DIR/models/Llama-3.1-8B-Instruct
export OUTPUT_DIR=$PROJECT_DIR/notebooks/prompt_engineering/output

mkdir -p "$OUTPUT_DIR"


echo "HOST=$(hostname)"
echo "START=$(date)"
echo "PY=$PY"


# Environment check
"$PY" -c "import torch; print('CUDA:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0))"


"$PY" -u notebooks/prompt_engineering/generate_prompts.py


echo "END=$(date)"


