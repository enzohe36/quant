#!/bin/bash

# nohup bash train_model.sh 2>&1 & echo "kill -- -$(ps -o pgid= -p $! | tr -d ' ')" | at 07:00 2026-03-18

# optuna-dashboard sqlite:///sweep.db --port 8080

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate py311

TRAIN_CMDS=(
  "python train_model.py"
  "python train_model.py --eval_all_peers true"
  # "python sweep.py --n_trials 200"
)

TIMESTAMP=$(date +%Y%m%d%H%M%S)

for TRAIN_CMD in "${TRAIN_CMDS[@]}"; do
  FLAG_SUFFIX=$(echo "$TRAIN_CMD" | grep -oP '\-\-\S+\s+\S+' | sed 's/--//;s/ /_/;s/^/_/' | tr -d '\n')
  LOG="training_${TIMESTAMP}${FLAG_SUFFIX}.log"

  if echo "$TRAIN_CMD" | grep -q 'sweep.py'; then
    ZIP_PATTERN="sweep_*${FLAG_SUFFIX}.zip"
  else
    ZIP_PATTERN="checkpoints_*${FLAG_SUFFIX}.zip"
  fi

  # Skip if a zip with matching flags already exists
  if ls $ZIP_PATTERN 1>/dev/null 2>&1; then
    echo "Skipping (zip exists): $TRAIN_CMD"
    continue
  fi

  $TRAIN_CMD > "$LOG" 2>&1 || { echo "Training failed: $TRAIN_CMD"; exit 1; }

  if echo "$TRAIN_CMD" | grep -q 'sweep.py'; then
    zip -r "sweep_${TIMESTAMP}${FLAG_SUFFIX}.zip" sweep_runs/ sweep.db sweep.py train_model.py feats_example.csv "$LOG" 2>/dev/null
    rm -rf sweep_runs sweep.db "$LOG"
  else
    zip -r "checkpoints_${TIMESTAMP}${FLAG_SUFFIX}.zip" checkpoints/ train_model.py feats_example.csv "$LOG" 2>/dev/null
    rm -rf checkpoints "$LOG"
  fi
done
