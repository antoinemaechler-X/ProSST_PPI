#!/bin/bash
#SBATCH --job-name=attention
#SBATCH --output=attention_%j.out
#SBATCH --error=attention_%j.err
#SBATCH --time=12:00:00
#SBATCH --mem=8G
#SBATCH --partition=gpu_test
#SBATCH --gres=gpu:1
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=antoinemaechler@fas.harvard.edu
#SBATCH --nice=10000

# Load your environment
source /n/sw/Mambaforge-23.11.0-0/etc/profile.d/conda.sh
conda activate gnn_ppi  # Make sure to use your correct environment name

# Go to your project directory
cd /n/netscratch/shakhnovich_lab/Lab/amaechler/ProSST_PPI

python MLP_attention_scores_new.py
