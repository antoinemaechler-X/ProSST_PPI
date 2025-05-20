#!/bin/bash
#SBATCH --job-name=preprocess_skemp
#SBATCH --output=preprocess_%j.out
#SBATCH --error=preprocess_%j.err
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --partition=shared,sapphire,shakhnovich
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=antoinemaechler@fas.harvard.edu
#SBATCH --nice=10000

# Load your environment
source /n/sw/Mambaforge-23.11.0-0/etc/profile.d/conda.sh
conda activate gnn_ppi  # Make sure to use your correct environment name

# Go to your project directory
cd /n/netscratch/shakhnovich_lab/Lab/amaechler/ProSST_PPI

# Run preprocessing with 16 workers
python MLP_ddG.py --preprocess_only --pdb_dir data/SKEMPI2/SKEMPI2_cache/wildtype --num_workers 8
