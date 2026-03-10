#!/bin/bash

# nohup bash train_model.sh >> "training_$(date +%Y%m%d%H%M%S).log" 2>&1 & echo "kill -- -$(ps -o pgid= -p $! | tr -d ' ')" | at 07:00 2026-03-02
#
# Monitor sweep progress:
#   optuna-dashboard sqlite:///sweep.db --port 8080

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate py311

TRAIN_CMD="python train_model.py"

FLAG_SUFFIX=$(echo "$TRAIN_CMD" | grep -oP '\-\-\S+' | sed 's/--//;s/=//;s/^/_/' | tr -d '\n')
TIMESTAMP=$(date +%Y%m%d%H%M%S)

$TRAIN_CMD || { echo "Training failed: $TRAIN_CMD"; exit 1; }

if echo "$TRAIN_CMD" | grep -q 'sweep.py'; then
  zip -r "sweep_${TIMESTAMP}${FLAG_SUFFIX}.zip" sweep_runs/ sweep.db sweep.py train_model.py feats_example.csv training_*.log 2>/dev/null
  rm -rf sweep_runs sweep.db training_*.log
else
  zip -r "checkpoints_${TIMESTAMP}${FLAG_SUFFIX}.zip" checkpoints/ train_model.py feats_example.csv training_*.log 2>/dev/null
  rm -rf checkpoints training_*.log
fi
