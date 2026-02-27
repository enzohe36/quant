#!/bin/bash

# nohup bash train_model.sh >> "training_$(date +%Y%m%d%H%M%S).log" 2>&1 & echo "kill -- -$(ps -o pgid= -p $! | tr -d ' ')" | at 07:00 2026-03-02
#
# Monitor sweep progress:
#   optuna-dashboard sqlite:///ppo_sweep.db --port 8080

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate py311

TRAIN_CMD="python train_model.py --sweep=true"

FLAG_SUFFIX=$(echo "$TRAIN_CMD" | grep -oP '\-\-\S+' | sed 's/--//;s/=//;s/^/_/' | tr -d '\n')
SCRIPT_NAME=$(echo "$TRAIN_CMD" | awk '{print $2}')
SCRIPT_BASE=$(basename "$SCRIPT_NAME" .py)
IS_SWEEP=$(echo "$TRAIN_CMD" | grep -c '\-\-sweep')
TIMESTAMP=$(date +%Y%m%d%H%M%S)

$TRAIN_CMD || { echo "Training failed: $TRAIN_CMD"; exit 1; }

if [ "$IS_SWEEP" -gt 0 ]; then
  mv training_*.log sweep_runs/ 2>/dev/null
  cp "$SCRIPT_NAME" sweep_runs/
  zip -r "sweep_${TIMESTAMP}_${SCRIPT_BASE}${FLAG_SUFFIX}.zip" sweep_runs/ ppo_sweep.db
  rm -rf sweep_runs ppo_sweep.db
else
  mv training_*.log checkpoints/ 2>/dev/null
  cp "$SCRIPT_NAME" example.csv checkpoints/
  zip -r "checkpoints_${TIMESTAMP}_${SCRIPT_BASE}${FLAG_SUFFIX}.zip" checkpoints/
  rm -rf checkpoints
fi
