#!/bin/bash

set -euo pipefail

# Set to 10 for testing, or "all" for the full dataset.
OBSERVATION_LIMIT="all"

PROJECT_DIR="/rs1/researchers/a/amallav/species_classification/ecological_informatics"
PYTHON="/usr/local/usrapps/ftrscape/snair3/env_AiPipeline/bin/python"

cd "$PROJECT_DIR"

mkdir -p logs

LOG_FILE="logs/inat_fetch_${OBSERVATION_LIMIT}_$(date +%Y%m%d_%H%M%S).log"

echo "Host: $(hostname)"
echo "Started: $(date)"
echo "Observation limit: $OBSERVATION_LIMIT"

if [[ "$OBSERVATION_LIMIT" == "all" ]]; then
    "$PYTHON" -u data/run_observations.py 2>&1 | tee "$LOG_FILE"
else
    "$PYTHON" -u data/run_observations.py \
        --limit "$OBSERVATION_LIMIT" 2>&1 | tee "$LOG_FILE"
fi

echo "Finished: $(date)"
