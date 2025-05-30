import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import re

def normalize_id(pid):
    """Normalize protein ID by removing special characters and converting to lowercase."""
    return re.sub(r'[._]', '_', pid.strip().lower())

def read_predictions(file_path):
    """Read flexibility predictions from file."""
    predictions = {}
    current_protein = None
    current_values = []
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('>'):
                if current_protein is not None:
                    predictions[normalize_id(current_protein)] = np.array(current_values)
                current_protein = line[1:]
                current_values = []
            elif line:
                values = [float(x) for x in line.replace(', ', ',').split(',') if x]
                current_values.extend(values)
    if current_protein is not None:
        predictions[normalize_id(current_protein)] = np.array(current_values)
    return predictions

def calculate_differences(wt_pred, mt_pred, min_change_threshold=1e-5):
    """Calculate differences between WT and MT predictions for all mutations.
    For each mutation, finds the position with maximum absolute change and stores
    that change with its actual sign.
    Skips mutations where the maximum change is below min_change_threshold (unchanged chains)."""
    common_proteins = set(wt_pred.keys()) & set(mt_pred.keys())
    max_diffs = []  # Will store the actual value (not absolute) at position of max abs change
    mutation_info = []
    
    skipped = 0
    for protein in sorted(common_proteins):
        wt_values = wt_pred[protein]
        mt_values = mt_pred[protein]
        
        if len(wt_values) != len(mt_values):
            skipped += 1
            continue
            
        # Calculate differences
        diff = mt_values - wt_values
        # Find position of maximum absolute change
        max_abs_idx = np.argmax(np.abs(diff))
        max_abs_change = diff[max_abs_idx]
        
        # Skip if change is too small (unchanged chain)
        if abs(max_abs_change) < min_change_threshold:
            skipped += 1
        else:
            max_diffs.append(max_abs_change)
            mutation_info.append(protein)
    
    print(f"Analyzed all mutations:")
    print(f"- Included: {len(max_diffs)} mutations with significant changes")
    print(f"- Skipped: {skipped} mutations (unchanged chains or length mismatch)")
    
    return np.array(max_diffs), mutation_info

def plot_difference_distributions(max_diffs, mutation_info, output_dir='plots_flexibility'):
    """Create distribution plot for maximum flexibility changes."""
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    # Create single plot
    plt.figure(figsize=(10, 6))
    
    # Plot max differences
    sns.histplot(data=max_diffs, kde=True)
    plt.axvline(x=0, color='r', linestyle='--', alpha=0.5)
    plt.title('Distribution of Maximum Flexibility Changes\n(MT - WT) at Positions of Largest Change\n(Excluding Unchanged Chains)')
    plt.xlabel('Flexibility Change at Position of Maximum Absolute Change')
    plt.ylabel('Count')
    
    # Add statistics
    stats_text = (f'Mean: {np.mean(max_diffs):.3f}\n'
                 f'Std: {np.std(max_diffs):.3f}\n'
                 f'Min: {np.min(max_diffs):.3f}\n'
                 f'Max: {np.max(max_diffs):.3f}\n'
                 f'Positive Changes: {np.sum(max_diffs > 0)}/{len(max_diffs)}\n'
                 f'Negative Changes: {np.sum(max_diffs < 0)}/{len(max_diffs)}')
    plt.text(0.95, 0.95, stats_text, transform=plt.gca().transAxes,
             verticalalignment='top', horizontalalignment='right',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'max_flexibility_changes.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved plot to {output_dir}/max_flexibility_changes.png")

def main():
    # File paths
    wt_file = 'data/SKEMPI2/SKEMPI2_cache/skempi_output_wildtype-3D-all-predictions.txt'
    mt_file = 'data/SKEMPI2/SKEMPI2_cache/skempi_output-3D-all-predictions.txt'
    
    # Read predictions
    print("Reading predictions...")
    wt_predictions = read_predictions(wt_file)
    mt_predictions = read_predictions(mt_file)
    
    # Calculate differences (with threshold to skip unchanged chains)
    max_diffs, mutation_info = calculate_differences(wt_predictions, mt_predictions, 
                                                   min_change_threshold=1e-5)
    
    # Create plot
    plot_difference_distributions(max_diffs, mutation_info)
    
    print("Analysis complete!")

if __name__ == "__main__":
    main() 