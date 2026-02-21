#!/bin/bash

# nohup bash train_model.sh >> training.log 2>&1 &
# echo "kill -- -$(ps -o pgid= -p $! | tr -d ' ')" | at 07:00 2026-03-02

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate py311

python train_model.py --dropout=0.0 --market_dropout=0.5 >> training.log 2>&1
mv training.log checkpoints/
cp train_model.py example.csv checkpoints/

zip -r "checkpoints_$(date +%Y%m%d)_drop00_mdrop05.zip" checkpoints/
rm -rf checkpoints
