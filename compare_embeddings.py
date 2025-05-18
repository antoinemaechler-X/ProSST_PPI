import os
import glob
import numpy as np
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from tqdm import tqdm

def load_mutation_info(skempi_file='data/SKEMPI2/SKEMPI2.csv'):
    """Load mutation information from SKEMPI2.csv."""
    # Read CSV with comment character, but keep the # in column names
    df = pd.read_csv(skempi_file)
    # Create a mapping from PDB ID to mutations
    mutations = {}
    for _, row in df.iterrows():
        pdb_id = row['#Pdb_origin']
        full_id = row['#Pdb']  # Like '0_1CSE'
        mutation = row['Mutation(s)_cleaned']
        ddg = row['ddG']
        if pdb_id not in mutations:
            mutations[pdb_id] = {}
        mutations[pdb_id][full_id] = {'mutation': mutation, 'ddG': ddg}
    return mutations

def visualize_differences(wt_emb, opt_emb, base_name, mutation_info, save_dir=None):
    """Create visualizations of the differences between wildtype and optimized embeddings."""
    # Calculate absolute differences
    abs_diff = np.abs(opt_emb - wt_emb)
    
    # Create heatmap of absolute differences with capped values
    plt.figure(figsize=(20, 10))
    
    # Create a masked array for values > 1.0
    capped_diff = abs_diff.copy()
    high_values_mask = capped_diff > 1.0
    capped_diff[high_values_mask] = 1.0
    
    # Create custom colormap with red for high values
    cmap = plt.cm.viridis.copy()
    cmap.set_over('red')
    
    # Plot heatmap with special handling for values > 1.0
    sns.heatmap(capped_diff, cmap=cmap, vmin=0, vmax=1.0, robust=True)
    
    # Add text annotation for percentage of high values
    high_values_count = np.sum(high_values_mask)
    total_values = abs_diff.size
    high_values_percent = (high_values_count / total_values) * 100
    
    mutation_text = f"Mutation: {mutation_info['mutation']}, ddG: {mutation_info['ddG']:.2f}"
    high_values_text = f"\nValues > 1.0: {high_values_count:,} ({high_values_percent:.1f}%)"
    plt.title(f'Absolute Differences Heatmap for {base_name}\n{mutation_text}{high_values_text}')
    plt.xlabel('Embedding Dimension')
    plt.ylabel('Token Position')
    
    if save_dir:
        plt.savefig(os.path.join(save_dir, f'{base_name}_abs_diff_heatmap.png'))
        plt.close()
    else:
        plt.show()
        plt.close()

    # Distribution of absolute differences with special handling for values > 1.0
    plt.figure(figsize=(10, 5))
    
    # Create histogram data with clipped values
    diff_data = abs_diff.flatten()
    hist_data = diff_data[diff_data <= 1.0]  # Only plot regular histogram for values <= 1.0
    
    # Plot main histogram
    sns.histplot(hist_data, bins=50)
    
    # Add count of values > 1.0 as a single bar
    high_values_count = np.sum(diff_data > 1.0)
    if high_values_count > 0:
        max_height = plt.gca().get_ylim()[1]
        plt.bar(1.0, high_values_count, width=0.05, color='red', alpha=0.7, 
                label=f'Values > 1.0\n(n={high_values_count})')
        plt.legend()
    
    plt.title(f'Distribution of Absolute Differences for {base_name}\n{mutation_text}')
    plt.xlabel('Absolute Difference')
    plt.ylabel('Count')
    
    if save_dir:
        plt.savefig(os.path.join(save_dir, f'{base_name}_abs_diff_dist.png'))
        plt.close()
    else:
        plt.show()
        plt.close()

def compare_embeddings(wildtype_dir, optimized_dir, protein_id, save_plots=False, full=False):
    """Compare embeddings between wildtype and optimized directories for a specific protein."""
    # Create plots directory if saving plots
    plots_dir = None
    if save_plots:
        plots_dir = os.path.join(os.path.dirname(wildtype_dir), 'embedding_comparison_plots')
        os.makedirs(plots_dir, exist_ok=True)
    
    # Load mutation information
    mutations = load_mutation_info()
    if protein_id not in mutations:
        print(f"No mutation information found for protein {protein_id}")
        return []
    
    # Get all wildtype embedding files for this protein
    wildtype_files = []
    if full:
        pattern = os.path.join(wildtype_dir, f"*_{protein_id}_full_embeddings.npy")
        wildtype_files = glob.glob(pattern)
    else:
        for chain in ['A', 'B']:
            pattern = os.path.join(wildtype_dir, f"*_{protein_id}_{chain}_embeddings.npy")
            chain_files = glob.glob(pattern)
            if chain_files:
                wildtype_files.extend(chain_files)
    
    if not wildtype_files:
        print(f"No .npy files found for protein {protein_id} in wildtype directory: {wildtype_dir}")
        if full:
            print(f"Looked for pattern: *_{protein_id}_full_embeddings.npy")
        else:
            print("Looked for patterns: *_{protein_id}_A_embeddings.npy and *_{protein_id}_B_embeddings.npy")
        return []
    
    print(f"Found {len(wildtype_files)} files for protein {protein_id}")
        
    results = []
    
    for wt_path in tqdm(wildtype_files, desc=f"Processing {protein_id} mutations"):
        try:
            # Get base name and construct optimized path
            base_name = os.path.basename(wt_path).replace('_embeddings.npy', '')
            opt_path = os.path.join(optimized_dir, f"{base_name}_embeddings.npy")
            
            # Get mutation info
            parts = base_name.split('_')
            prefix = parts[0]
            full_id = f"{prefix}_{protein_id}"
            
            if full_id not in mutations[protein_id]:
                print(f"No mutation information found for {full_id}")
                continue
            mutation_info = mutations[protein_id][full_id]
            
            if not os.path.exists(opt_path):
                print(f"Skipping {base_name}: No matching optimized embedding found")
                continue
                
            # Load embeddings
            wt_emb = np.load(wt_path)
            opt_emb = np.load(opt_path)
            
            if wt_emb.shape != opt_emb.shape:
                print(f"Skipping {base_name}: Shape mismatch - WT: {wt_emb.shape}, OPT: {opt_emb.shape}")
                continue
            
            # Calculate differences
            abs_diff = np.abs(opt_emb - wt_emb)
            
            # Calculate statistics
            significant_changes = np.sum(abs_diff > 0.1)
            max_abs_diff = float(np.max(abs_diff))
            mean_abs_diff = float(np.mean(abs_diff))
            
            # Find top 10 most different tokens
            token_diffs = np.sum(abs_diff, axis=1)  # Sum across embedding dimensions
            top_tokens = np.argsort(-token_diffs)[:10]  # Top 10 most different tokens
            
            # Create visualizations if requested
            if save_plots:
                visualize_differences(wt_emb, opt_emb, base_name, mutation_info, plots_dir)
            
            results.append({
                'name': base_name,
                'mutation': mutation_info['mutation'],
                'ddG': mutation_info['ddG'],
                'significant_changes': significant_changes,
                'max_abs_diff': max_abs_diff,
                'mean_abs_diff': mean_abs_diff,
                'top_tokens': top_tokens,
                'top_token_diffs': token_diffs[top_tokens],
                'shape': wt_emb.shape
            })
            
        except Exception as e:
            print(f"Error processing {wt_path}: {str(e)}")
            continue
    
    return results

def main():
    parser = argparse.ArgumentParser(description='Compare embeddings between wildtype and optimized versions')
    parser.add_argument('protein_id', type=str, 
                        help='Protein ID to analyze (e.g., 1CSE)')
    parser.add_argument('--wildtype_dir', type=str, 
                        default='data/SKEMPI2/SKEMPI2_cache/embedding_wildtype_full_2048',
                        help='Directory containing wildtype embeddings')
    parser.add_argument('--optimized_dir', type=str, 
                        default='data/SKEMPI2/SKEMPI2_cache/embedding_optimized_full_2048',
                        help='Directory containing optimized embeddings')
    parser.add_argument('--save_plots', action='store_true',
                        help='Save visualization plots to disk')
    parser.add_argument('--full', action='store_true',
                        help='Use full embeddings (combined chains) instead of separate chain embeddings')
    args = parser.parse_args()
    
    results = compare_embeddings(args.wildtype_dir, args.optimized_dir, 
                               args.protein_id, args.save_plots, args.full)
    
    if not results:
        print("No matching embedding pairs found!")
        return
    
    # Sort by total absolute difference
    results.sort(key=lambda x: x['significant_changes'], reverse=True)
    
    # Print results
    print(f"\nAnalyzed {len(results)} embedding pairs for protein {args.protein_id}")
    print("\nResults (sorted by number of significant changes):")
    for i, result in enumerate(results):
        print(f"\n{i+1}. {result['name']}")
        print(f"   Mutation: {result['mutation']}, ddG: {result['ddG']:.2f}")
        print(f"   Dimensions with change > 0.1: {result['significant_changes']}")
        print(f"   Max change: {result['max_abs_diff']:.4f}")
        print(f"   Mean change: {result['mean_abs_diff']:.4f}")
        print(f"   Top 10 most different tokens (index, total difference):")
        for token_idx, diff in zip(result['top_tokens'], result['top_token_diffs']):
            print(f"     Token {token_idx}: {diff:.4f}")

if __name__ == "__main__":
    main() 