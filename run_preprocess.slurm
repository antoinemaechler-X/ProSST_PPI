#!/bin/bash
#SBATCH -p gpu_test
#SBATCH --gres=gpu:1
#SBATCH --mem=50G
#SBATCH -t 12:00:00
#SBATCH -o logs/preprocess_out_%j.log
#SBATCH -e logs/preprocess_err_%j.log
#SBATCH --job-name=ppi_preprocess

# OPTIONAL: load only if you ever needed to before
# module load python/3.11.5
module load cuda/11.8.0-fasrc01

# Activate your existing venv (assumes you made it without loading a Python module)
source ~/ppi-env/bin/activate

cd ~/PPI
python PPI_preprocess.py
