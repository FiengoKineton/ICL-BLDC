#!/bin/bash
#SBATCH --job-name=icl_gpu_test
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=2-00:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --partition=gpu-light

module purge
module load cuda/12.1
module load python/3.10

cd /home/giuseppe_fiengo/Sweep_Speed_Estm
mkdir -p logs

python3 train_zerostep.py
