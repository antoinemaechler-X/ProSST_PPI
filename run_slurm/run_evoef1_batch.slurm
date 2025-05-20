#!/bin/bash
#SBATCH --job-name=evoef1_M1340    # a short name for your job
#SBATCH --output=evoef1_%j.out      # STDOUT → evoef1_<jobid>.out
#SBATCH --error=evoef1_%j.err       # STDERR → evoef1_<jobid>.err
#SBATCH --time=48:00:00             # Max walltime (HH:MM:SS)
#SBATCH --cpus-per-task=4           # cores per task
#SBATCH --mem=8G                    # total RAM
#SBATCH --partition=shared,sapphire,shakhnovich
#SBATCH --nice=10000                # lower priority
#SBATCH --mail-type=END,FAIL        # send email on job end/fail
#SBATCH --mail-user=antoinemaechler@fas.harvard.edu

# === load your environment ===
source /n/sw/Mambaforge-23.11.0-0/etc/profile.d/conda.sh
conda activate RDE

# === move into your working folder ===
cd /n/netscratch/shakhnovich_lab/Lab/amaechler/DDAffinity_up

#cd ~/DDAffinity/rde/datasets
cd rde/datasets

# === run the EvoEF1 generation script ===
# note: make sure PDB_generate_evoef1.py is executable or call python explicitly
python PDB_generate_evoef1.py

# === end of script ===
