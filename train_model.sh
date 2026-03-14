#!/bin/bash

# nohup bash train_model.sh >> "training_$(date +%Y%m%d%H%M%S).log" 2>&1 & echo "kill -- -$(ps -o pgid= -p $! | tr -d ' ')" | at 07:00 2026-03-16

# optuna-dashboard sqlite:///sweep.db --port 8080

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate py311

TRAIN_CMDS=(
  "python train_model.py"
  "python train_model.py --self_dropout 0.5"
  "python train_model.py --self_dropout 1.0"
  "python train_model.py --pool_self true"
  "python train_model.py --pool_self true --eval_all_peers true"
  # "python sweep.py --n_trials 200"
)

TIMESTAMP=$(date +%Y%m%d%H%M%S)

for TRAIN_CMD in "${TRAIN_CMDS[@]}"; do
  FLAG_SUFFIX=$(echo "$TRAIN_CMD" | grep -oP '\-\-\S+\s+\S+' | sed 's/--//;s/ /_/;s/^/_/' | tr -d '\n')

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

  $TRAIN_CMD || { echo "Training failed: $TRAIN_CMD"; exit 1; }

  if echo "$TRAIN_CMD" | grep -q 'sweep.py'; then
    zip -r "sweep_${TIMESTAMP}${FLAG_SUFFIX}.zip" sweep_runs/ sweep.db sweep.py train_model.py feats_example.csv training_*.log 2>/dev/null
    rm -rf sweep_runs sweep.db training_*.log
  else
    zip -r "checkpoints_${TIMESTAMP}${FLAG_SUFFIX}.zip" checkpoints/ train_model.py feats_example.csv training_*.log 2>/dev/null
    rm -rf checkpoints training_*.log
  fi
done
