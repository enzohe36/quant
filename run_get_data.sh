#!/bin/bash
conda activate /opt/anaconda3/envs/myenv
cd ~/Documents/quant
Rscript get_data.r
Rscript get_data.r
zip -r "data_$(date +%Y%m%d).zip" data
