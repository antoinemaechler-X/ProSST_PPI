#!/bin/bash
#SBATCH --job-name=foldx_skemp
#SBATCH --output=foldx_%j.out
#SBATCH --error=foldx_%j.err
#SBATCH --time=48:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --partition=shared,sapphire,shakhnovich
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=antoinemaechler@fas.harvard.edu
#SBATCH --nice=10000

# Load your environment
source /n/sw/Mambaforge-23.11.0-0/etc/profile.d/conda.sh
conda activate RDE

# Go to your project directory
cd /n/netscratch/shakhnovich_lab/Lab/amaechler/DDAffinity_up

#cd ~/DDAffinity/rde/datasets
cd rde/datasets

# Run your script
python PDB_generate_new_version.py
