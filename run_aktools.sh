#!/bin/bash
conda activate /opt/anaconda3/envs/myenv
pip install aktools --upgrade -i https://pypi.org/simple
pip install akshare --upgrade -i https://pypi.org/simple
python -m aktools
