import os
import glob
import matplotlib.pyplot as plt
from collections import Counter
from matplotlib.patches import Patch

def plot_structure_distribution(folder_path, K):
    # Grab all .fasta files
    fasta_files = glob.glob(os.path.join(folder_path, "*.fasta"))
    print(f"Found {len(fasta_files)} fasta files in {folder_path}")

    total_counts = Counter()

    for fasta_file in fasta_files:
        with open(fasta_file, "r") as f:
            lines = f.readlines()
            if len(lines) < 2:
                continue
            sequence_line = lines[1].strip()
            try:
                numbers = list(map(int, sequence_line.split(",")))
                total_counts.update(numbers)
            except ValueError:
                print(f"Skipping {fasta_file} (invalid format)")

    # Create full histogram and proportions
    total = sum(total_counts.values())
    histogram = [total_counts.get(i, 0) for i in range(1, K + 1)]
    proportions = [count / total for count in histogram]

    # --- Plot 1: Raw count histogram ---
    plt.figure(figsize=(10, 4))
    plt.bar(range(1, K + 1), histogram)
    plt.xlabel("Structure token (1 to K)")
    plt.ylabel("Frequency")
    plt.title(f"Raw structure token frequency in {folder_path}")
    plt.tight_layout()
    plt.grid(True)
    plt.savefig(os.path.join(folder_path, f"structure_hist_K{K}.png"))
    print(f"Raw count histogram saved to {folder_path}")

        # --- Plot 2: Proportion bins ---
    bins = {
        ">5%": 0,
        "2–5%": 0,
        "1–2%": 0,
        "0.1–1%": 0,
        "<0.1%": 0,
        "Zero": 0
    }

    bin_labels = list(bins.keys())
    bin_colors = ["darkred", "orange", "gold", "skyblue", "lightgray", "black"]
    token_bins = []

    for p in proportions:
        if p == 0:
            bins["Zero"] += 1
            token_bins.append("Zero")
        elif p > 0.05:
            bins[">5%"] += 1
            token_bins.append(">5%")
        elif p > 0.02:
            bins["2–5%"] += 1
            token_bins.append("2–5%")
        elif p > 0.01:
            bins["1–2%"] += 1
            token_bins.append("1–2%")
        elif p > 0.001:
            bins["0.1–1%"] += 1
            token_bins.append("0.1–1%")
        else:
            bins["<0.1%"] += 1
            token_bins.append("<0.1%")

    # Plot with counts above bars
    plt.figure(figsize=(8, 4))
    bar_values = [bins[b] for b in bin_labels]
    bars = plt.bar(bin_labels, bar_values, color=bin_colors)
    for bar, value in zip(bars, bar_values):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, str(value),
                 ha='center', va='bottom', fontsize=9)

    plt.ylabel("Number of tokens")
    plt.xlabel("Proportion of total counts")
    plt.title(f"Token frequency bins in {folder_path}")
    plt.tight_layout()
    plt.savefig(os.path.join(folder_path, f"structure_bins_K{K}.png"))
    print(f"Binned proportion plot saved to {folder_path}")

    # --- Plot 3: Color band of token categories (as colored bar) ---
    plt.figure(figsize=(12, 4))
    color_map = dict(zip(bin_labels, bin_colors))
    colors = [color_map[b] for b in token_bins]
    plt.bar(range(len(colors)), [1] * len(colors), color=colors, width=1)
    plt.xticks([])
    plt.yticks([])
    plt.title(f"Structure token category band (1–{K})")
    # Add color legend
    legend_patches = [Patch(facecolor=color_map[label], label=label) for label in bin_labels]
    plt.legend(handles=legend_patches, loc='upper center', bbox_to_anchor=(0.5, -0.6),
            ncol=len(bin_labels), fontsize=8, frameon=False)
    plt.tight_layout()
    plt.savefig(os.path.join(folder_path, f"structure_band_K{K}.png"))
    print(f"Color band saved to {folder_path}")


plot_structure_distribution("PPI_data/2048", 2048)
