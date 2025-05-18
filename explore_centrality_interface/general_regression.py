#!/usr/bin/env python3
import os
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from scipy.stats import pearsonr, spearmanr

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.model_selection import GroupKFold, cross_val_predict

from burial_score import burial_score
from interface_score import interface_score

def get_first_N_proteins(csv_path, N):
    """
    Return the first N unique protein codes from '#Pdb_origin' in the CSV.
    """
    df = pd.read_csv(csv_path)
    proteins = df['#Pdb_origin'].dropna().unique()
    return proteins[:N]

def load_single_point_data(csv_path, pdb_dir, proteins,
                           neighbor_count, sigma_interface):
    """
    For each protein in `proteins`, load all *single*-mutation rows,
    compute burial and interface scores, and return X, y, and protein IDs.
    """
    X1, X2, y_vals, prot_ids = [], [], [], []
    for p in proteins:
        df = pd.read_csv(csv_path)
        df = df[df['#Pdb_origin'] == p]
        df = df[pd.to_numeric(df['ddG'], errors='coerce').notnull()]

        for _, row in df.iterrows():
            muts = row['Mutation(s)_cleaned'].strip('"').split(',')
            if len(muts) != 1:
                continue  # only single‐point mutations
            mut = muts[0]
            ddg_val = abs(float(row['ddG']))
            pdb_file = os.path.join(pdb_dir, f"{row['#Pdb']}.pdb")

            b = burial_score(pdb_file, mut, neighbor_count=neighbor_count)
            i = interface_score(pdb_file, mut, sigma_interface=sigma_interface)

            X1.append(b)
            X2.append(i)
            y_vals.append(ddg_val)
            prot_ids.append(p)

    X = np.column_stack([X1, X2])
    return X, np.array(y_vals), np.array(prot_ids)

def main():
    parser = argparse.ArgumentParser(
        description="Global regression across complexes with group K-fold CV"
    )
    parser.add_argument('-N', '--num_proteins', type=int, required=True,
                        help='Number of complexes to include')
    parser.add_argument('--csv', type=str, required=True,
                        help='Path to SKEMPI2 CSV')
    parser.add_argument('--pdb_dir', type=str, required=True,
                        help='Directory containing wildtype PDB files')
    parser.add_argument('-k', '--neighbors', type=int, default=9,
                        help='k for burial score (default:9)')
    parser.add_argument('-s', '--sigma_interface', type=float, default=1.0,
                        help='σ for interface score (default:1.0)')
    parser.add_argument('-d', '--degree', type=int, choices=[1,2],
                        default=1, help='Regression degree (1 or 2)')
    parser.add_argument('--cv', type=int, default=10,
                        help='Number of folds for group K-fold CV (default:10)')
    parser.add_argument('--random_state', type=int, default=42,
                        help='Random seed for CV splitting')
    args = parser.parse_args()

    # 1) select first N complexes
    proteins = get_first_N_proteins(args.csv, args.num_proteins)

    # 2) load all single‐point mutations for those complexes
    X, y, prot_ids = load_single_point_data(
        args.csv, args.pdb_dir, proteins,
        neighbor_count=args.neighbors,
        sigma_interface=args.sigma_interface
    )

    # 3) exclude complexes with fewer than 3 mutations
    counts = pd.Series(prot_ids).value_counts()
    valid_prots = counts[counts >= 3].index.tolist()
    mask = np.isin(prot_ids, valid_prots)
    X, y, prot_ids = X[mask], y[mask], prot_ids[mask]
    proteins = valid_prots  # update group list

    # 4) build pipeline (identity or polynomial)
    if args.degree == 2:
        pipeline = Pipeline([
            ('poly', PolynomialFeatures(degree=2, include_bias=False)),
            ('lr',   LinearRegression())
        ])
    else:
        pipeline = Pipeline([
            ('id', FunctionTransformer(lambda x: x, validate=False)),
            ('lr', LinearRegression())
        ])

    # 5) group K-fold cross-validation
    n_splits = min(args.cv, len(proteins))
    group_kfold = GroupKFold(n_splits=n_splits)
    y_pred = cross_val_predict(
        pipeline, X, y,
        groups=prot_ids,
        cv=group_kfold
    )

    # 6) compute metrics on all out-of-group predictions
    r2    = np.nan_to_num(pearsonr(y, y_pred)[0]**2, nan=0.0)  # r² from Pearson
    pr, _ = pearsonr(y, y_pred)
    sr, _ = spearmanr(y, y_pred)

    # 7) plot true vs predicted (all points)
    fig, ax = plt.subplots()
    ax.scatter(y, y_pred, alpha=0.7)
    mn, mx = min(y.min(), y_pred.min()), max(y.max(), y_pred.max())
    ax.plot([mn, mx], [mn, mx], 'k--', lw=1)
    ax.set_xlabel('True Absolute ddG')
    ax.set_ylabel('Predicted Absolute ddG')
    ax.set_title(
        f"{len(proteins)} complexes | {n_splits}-fold GroupCV | "
        f"deg={args.degree}, k={args.neighbors}, σ={args.sigma_interface}"
    )
    metrics = f"R²={r2:.2f}\nPearson={pr:.2f}\nSpearman={sr:.2f}"
    ax.text(0.05, 0.95, metrics, transform=ax.transAxes,
            fontsize=8, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))

    # 8) save plot
    out_dir = os.path.join(os.path.dirname(__file__), 'plots_centrality_interface')
    os.makedirs(out_dir, exist_ok=True)
    out_fn = (
        f"GroupCV_{len(proteins)}prot_{n_splits}fold_"
        f"deg{args.degree}_k{args.neighbors}_s{args.sigma_interface}.png"
    )
    out_path = os.path.join(out_dir, out_fn)
    plt.tight_layout()
    plt.savefig(out_path)
    print(f"Saved {out_path}")

if __name__ == '__main__':
    main()
