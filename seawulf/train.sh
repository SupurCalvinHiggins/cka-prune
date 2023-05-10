#!/bin/bash
#SBATCH -t 30:00:00
#SBATCH --nodes=1 --ntasks-per-node=6
#SBATCH --export=NONE

module load Python/3.9.6-GCCcore-11.2.0
pip install --user torch
pip install --user numpy
pip install --user seaborn
pip install --user torchvision
python -u main_train.py $1
