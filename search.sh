#!/bin/bash
#SBATCH -t 10:00:00
#SBATCH --nodes=1 --ntasks-per-node=4
#SBATCH --export=NONE

source /etc/profile.d/modules.sh
module load Python/3.9.6-GCCcore-11.2.0
#module load PyTorch/1.12.1-foss-2022a-CUDA-11.7.0
pip install --user torch
pip install --user numpy
pip install --user seaborn
pip install --user torchvision
pip install --user wandb
python -u main_search.py $1
