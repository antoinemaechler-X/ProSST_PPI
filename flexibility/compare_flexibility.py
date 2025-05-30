import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import re

def normalize_id(pid):
    return re.sub(r'[._]', '_', pid.strip().lower())

def read_predictions(file_path):
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
                # Accept both comma and comma+space
                values = [float(x) for x in line.replace(', ', ',').split(',') if x]
                current_values.extend(values)
    if current_protein is not None:
        predictions[normalize_id(current_protein)] = np.array(current_values)
    return predictions

def calculate_correlations(ml_pred, enm_pred):
    common_proteins = set(ml_pred.keys()) & set(enm_pred.keys())
    correlations = {}
    for protein in sorted(common_proteins):
        ml_values = ml_pred[protein]
        enm_values = enm_pred[protein]
        if len(ml_values) != len(enm_values):
            print(f"Skipping {protein}: ML={len(ml_values)}, ENM={len(enm_values)}")
            continue
        pearson_r, pearson_p = stats.pearsonr(ml_values, enm_values)
        spearman_r, spearman_p = stats.spearmanr(ml_values, enm_values)
        correlations[protein] = {
            'pearson_r': pearson_r,
            'pearson_p': pearson_p,
            'spearman_r': spearman_r,
            'spearman_p': spearman_p,
            'ml_values': ml_values,
            'enm_values': enm_values
        }
    return correlations, sorted(correlations.keys())

def save_results_csv(correlations, output_file='correlation_results.csv'):
    rows = []
    for protein, corr in correlations.items():
        rows.append({
            'Protein': protein,
            'Pearson_r': corr['pearson_r'],
            'Pearson_p': corr['pearson_p'],
            'Spearman_r': corr['spearman_r'],
            'Spearman_p': corr['spearman_p']
        })
    df = pd.DataFrame(rows)
    df.to_csv(output_file, index=False)

def plot_comparison(correlations, protein_list, output_file='prediction_comparison.png'):
    # Global scatter
    all_ml = np.concatenate([correlations[p]['ml_values'] for p in protein_list])
    all_enm = np.concatenate([correlations[p]['enm_values'] for p in protein_list])
    plt.figure(figsize=(8, 8))
    sns.scatterplot(x=all_ml, y=all_enm, alpha=0.5, s=20)
    min_val = min(all_ml.min(), all_enm.min())
    max_val = max(all_ml.max(), all_enm.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5)
    plt.xlabel('ML Flexibility Prediction')
    plt.ylabel('ENM Flexibility Prediction')
    plt.title('ML vs ENM Flexibility Predictions (all proteins)')
    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
    plt.close()
    # Per-protein grid
    n = len(protein_list)
    ncols = 3
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5*ncols, 5*nrows))
    axes = axes.flatten()
    for i, protein in enumerate(protein_list):
        ax = axes[i]
        ml = correlations[protein]['ml_values']
        enm = correlations[protein]['enm_values']
        ax.scatter(ml, enm, alpha=0.5, s=10)
        ax.plot([ml.min(), ml.max()], [ml.min(), ml.max()], 'k--', alpha=0.3)
        ax.set_title(protein)
        ax.set_xlabel('ML')
        ax.set_ylabel('ENM')
    for j in range(i+1, len(axes)):
        axes[j].axis('off')
    plt.tight_layout()
    plt.savefig('per_protein_comparison.png', dpi=200)
    plt.close()

def main():
    ml_predictions = read_predictions('test_output-3D-all-predictions.txt')
    enm_predictions = read_predictions('test_output-3D-all-predictions_enm.txt')
    correlations, protein_list = calculate_correlations(ml_predictions, enm_predictions)
    if not correlations:
        print("No valid protein pairs to compare. Exiting.")
        return
    print("\nCorrelation Analysis:")
    print("-" * 80)
    print(f"{'Protein':<15} {'Pearson r':>10} {'Pearson p':>12} {'Spearman r':>12} {'Spearman p':>12}")
    print("-" * 80)
    for protein in protein_list:
        corr = correlations[protein]
        print(f"{protein:<15} {corr['pearson_r']:>10.3f} {corr['pearson_p']:>12.2e} "
              f"{corr['spearman_r']:>12.3f} {corr['spearman_p']:>12.2e}")
    avg_pearson = np.mean([correlations[p]['pearson_r'] for p in protein_list])
    avg_spearman = np.mean([correlations[p]['spearman_r'] for p in protein_list])
    print("-" * 80)
    print(f"{'Average':<15} {avg_pearson:>10.3f} {'':>12} {avg_spearman:>12.3f}")
    save_results_csv(correlations)
    print("\nCSV results saved as 'correlation_results.csv'")
    plot_comparison(correlations, protein_list)
    print("Modern graphs saved as 'prediction_comparison.png' and 'per_protein_comparison.png'")

if __name__ == "__main__":
    main()