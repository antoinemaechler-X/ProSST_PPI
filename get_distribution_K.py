import os
import glob
import matplotlib.pyplot as plt
from collections import Counter

def plot_structure_distribution(folder_path, K):
    # Grab all .fasta files
    fasta_files = glob.glob(os.path.join(folder_path, "*.fasta"))
    print(f"Found {len(fasta_files)} fasta files in {folder_path}")

    total_counts = Counter()

    for fasta_file in fasta_files:
        with open(fasta_file, "r") as f:
            lines = f.readlines()
            if len(lines) < 2:
                continue  # skip malformed files
            sequence_line = lines[1].strip()
            try:
                numbers = list(map(int, sequence_line.split(",")))
                total_counts.update(numbers)
            except ValueError:
                print(f"⚠️ Skipping {fasta_file} (invalid format)")

    # Fill in missing numbers (0 to K)
    histogram = [total_counts.get(i, 0) for i in range(1, K + 1)]

    plt.figure(figsize=(10, 5))
    plt.bar(range(1, K + 1), histogram)
    plt.xlabel("Structure token (1 to K)")
    plt.ylabel("Frequency")
    plt.title(f"Distribution of structure tokens in {folder_path}")
    plt.grid(True)
    plt.tight_layout()
    
    output_path = os.path.join(folder_path, f"structure_hist_K{K}.png")
    plt.savefig(output_path)
    print(f"Histogram saved to: {output_path}")

# Example usage:
plot_structure_distribution("PPI_data/2048", 2048)
