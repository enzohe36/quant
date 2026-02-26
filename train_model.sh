#!/bin/bash

# nohup bash train_model.sh >> training.log 2>&1 &
# echo "kill -- -$(ps -o pgid= -p $! | tr -d ' ')" | at 07:00 2026-03-02

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate py311

JOBS=(
  "python train_model.py --market_dropout=0.5"
  "python train_model_beta.py --market_dropout=0.5"
)

for TRAIN_CMD in "${JOBS[@]}"; do
  FLAG_SUFFIX=$(echo "$TRAIN_CMD" | grep -oP '\-\-\S+' | sed 's/--//;s/=//;s/^/_/' | tr -d '\n')

  SCRIPT_NAME=$(echo "$TRAIN_CMD" | awk '{print $2}')

  $TRAIN_CMD >> training.log 2>&1
  mv training.log checkpoints/
  cp "$SCRIPT_NAME" example.csv checkpoints/

  SCRIPT_BASE=$(basename "$SCRIPT_NAME" .py)

  zip -r "checkpoints_$(date +%Y%m%d)_${SCRIPT_BASE}${FLAG_SUFFIX}.zip" checkpoints/
  rm -rf checkpoints
done
