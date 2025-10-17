#!/bin/bash
conda activate /opt/anaconda3/envs/myenv
cd ~/Documents/quant
Rscript get_data.r
Rscript get_data.r
ts=$(date +%Y%m%d)
zip -rq "data_$ts.zip" data/
mv data_$ts.zip ~/Library/CloudStorage/GoogleDrive-enzohe36@gmail.com/My\ Drive/quant/
