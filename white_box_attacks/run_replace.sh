#!/bin/sh


module load anaconda/3.0

#SBATCH --nodes = 2
#SBATCH --ntasks-per-node = 1
#SBATCH --TIME = 2-06:60:60
#SBATCH --mem = 4000
#SBATCH --gpus=<2>

python3 adv_training.py
