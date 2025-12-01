#!/bin/bash
#SBATCH --job-name=icl_cpu_test
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=02:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --partition=cpu-only

module purge
module load python/3.12

cd /home/your_user/Sweep_Speed_Estm
mkdir -p logs

python -u main.py