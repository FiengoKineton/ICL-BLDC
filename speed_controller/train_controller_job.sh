#!/bin/bash
#SBATCH --job-name=ST_contr_train
#SBATCH --output=output.txt
#SBATCH --error=error.txt
#SBATCH --time=0:05:00
#SBATCH --mem=16G
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1

module load python/3.10
module load cuda/12.1

python3 train_zerostep_control_HCP.py