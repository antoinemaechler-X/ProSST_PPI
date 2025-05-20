#!/bin/bash
#SBATCH --job-name=compute_scores
#SBATCH --output=compute_scores_%j.out
#SBATCH --error=compute_scores_%j.err
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
python #!/bin/bash
#SBATCH --job-name=prepro_score
#SBATCH --output=prepro_score_%j.out
#SBATCH --error=prepro_score_%j.err
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
python #!/bin/bash
#SBATCH --job-name=prepro_score
#SBATCH --output=prepro_score_%j.out
#SBATCH --error=prepro_score_%j.err
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
python compute_interface_burial_scores.py

