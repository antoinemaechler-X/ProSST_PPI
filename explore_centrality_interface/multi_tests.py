#!/usr/bin/env python3
import os
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from scipy.stats import pearsonr, spearmanr
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.metrics import r2_score
from tqdm import tqdm

from burial_score import burial_score, burial_scores
from interface_score import interface_score, interface_scores

def get_first_N_proteins(csv_path, N):
    """
    Return the first N unique protein codes from '#Pdb_origin' in the CSV.
    """
    df = pd.read_csv(csv_path)
    return df['#Pdb_origin'].dropna().unique().tolist()[:N]

def regress_and_plot_multiple(N, csv_path, pdb_dir,
                              mode='mean',
                              neighbors=9, sigma_interface=1.0,
                              degree=1, distinguish=False,
                              normalize=False):
    """
    Run regression on combined data from the first N proteins using 5-fold CV
    and plot real vs. predicted, with R², Pearson and Spearman shown in a box.
    Only single-point mutations are included.
    If normalize=True, scale ddG to [0,1].
    """
    proteins = get_first_N_proteins(csv_path, N)
    X1, X2, y_list, lengths = [], [], [], []

    # iterate with progress bar over proteins
    for p in tqdm(proteins, desc="Processing proteins"):
        df = pd.read_csv(csv_path)
        df = df[df['#Pdb_origin'] == p]
        # drop rows where ddG is missing or non-numeric
        df = df[pd.to_numeric(df['ddG'], errors='coerce').notnull()]

        # iterate rows
        for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Mutations in {p}", leave=False):
            muts = row['Mutation(s)_cleaned'].strip('"').split(',')
            # only single-point mutations
            if len(muts) != 1:
                continue

            ddg = abs(float(row['ddG']))
            pdb_file = os.path.join(pdb_dir, f"{row['#Pdb']}.pdb")

            b_list = burial_scores(pdb_file, muts, neighbor_count=neighbors)
            i_list = interface_scores(pdb_file, muts, sigma_interface=sigma_interface)

            if mode == 'mean':
                b, i = np.mean(b_list), np.mean(i_list)
            elif mode == 'sum':
                b, i = np.sum(b_list), np.sum(i_list)
            else:  # max
                b, i = np.max(b_list), np.max(i_list)

            X1.append(b)
            X2.append(i)
            y_list.append(ddg)
            lengths.append(1)

    # convert to arrays
    X = np.column_stack([X1, X2])
    y = np.array(y_list)
    lengths = np.array(lengths)

    # normalize ddG if requested
    if normalize:
        y_min, y_max = y.min(), y.max()
        if y_max > y_min:
            y = (y - y_min) / (y_max - y_min)
        else:
            y = np.zeros_like(y)

    # build pipeline: identity or polynomial
    if degree == 2:
        estimator = Pipeline([
            ('poly', PolynomialFeatures(degree=2, include_bias=False)),
            ('lr',   LinearRegression())
        ])
    elif degree == 3:
        estimator = Pipeline([
            ('poly', PolynomialFeatures(degree=3, include_bias=False)),
            ('lr',   LinearRegression())
        ])
    else:
        estimator = Pipeline([
            ('id', FunctionTransformer(lambda x: x, validate=False)),
            ('lr', LinearRegression())
        ])

    # 5-fold cross-validation
    print("Running 5-fold cross-validation...")
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    y_pred = cross_val_predict(estimator, X, y, cv=cv)

    # compute metrics
    r2 = r2_score(y, y_pred)
    pr, _ = pearsonr(y, y_pred)
    sr, _ = spearmanr(y, y_pred)

    # plot true vs predicted
    fig, ax = plt.subplots()
    ax.scatter(y, y_pred, alpha=0.7)

    mn, mx = min(y.min(), y_pred.min()), max(y.max(), y_pred.max())
    ax.plot([mn, mx], [mn, mx], 'k--', lw=1)
    ax.set_xlabel('True ddG' + (' (norm)' if normalize else ''))
    ax.set_ylabel('Predicted ddG' + (' (norm)' if normalize else ''))
    ax.set_title(
        f'{N} proteins | 5-fold CV | mode={mode}, k={neighbors}, '
        f'σ={sigma_interface}, deg={degree}' +
        (' | normalized' if normalize else '')
    )

    metrics = f"R²={r2:.2f}\nPearson={pr:.2f}\nSpearman={sr:.2f}"
    ax.text(0.05, 0.95, metrics, transform=ax.transAxes,
            fontsize=8, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))

    # save plot
    out_dir = os.path.join(os.path.dirname(__file__), 'plots_centrality_interface')
    os.makedirs(out_dir, exist_ok=True)
    fn = (f'Regression_{N}_prot_5fold_deg{degree}_k{neighbors}'
          f'_s{sigma_interface}' + ('_norm' if normalize else '') + '.png')
    out_path = os.path.join(out_dir, fn)
    plt.tight_layout()
    plt.savefig(out_path)
    print(f"Saved {out_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Run regression across multiple proteins with 5-fold CV"
    )
    parser.add_argument('-N', '--num_proteins', type=int, default=30,
                        help='Number of proteins to include (default: 30)')
    parser.add_argument('-m', '--mode',
                        choices=['mean', 'sum', 'max'], default='mean',
                        help="Aggregation mode (default: mean)")
    parser.add_argument('-k', '--neighbors', type=int, default=9,
                        help='k for burial score (default: 9)')
    parser.add_argument('-s', '--sigma-interface', dest='sigma_interface',
                        type=float, default=1.0,
                        help='σ for interface Gaussian (default: 1.0)')
    parser.add_argument('-d', '--degree', type=int, choices=[1, 2, 3],
                        default=1, help='Regression degree (1 or 2)')
    parser.add_argument('--distinguish', action='store_true', default=False,
                        help='Color‐code points by number of mutations (no effect, single only)')
    parser.add_argument('--normalize', action='store_true', default=False,
                        help='Normalize ddG to [0,1] before regression')
    parser.add_argument('--csv', type=str,
                        default='data/SKEMPI2/SKEMPI2.csv',
                        help='Path to CSV (default: data/SKEMPI2/SKEMPI2.csv)')
    parser.add_argument('--pdb_dir', type=str,
                        default='data/SKEMPI2/SKEMPI2_cache/wildtype',
                        help='Path to PDB directory')
    args = parser.parse_args()

    regress_and_plot_multiple(
        args.num_proteins,
        args.csv,
        args.pdb_dir,
        mode=args.mode,
        neighbors=args.neighbors,
        sigma_interface=args.sigma_interface,
        degree=args.degree,
        distinguish=args.distinguish,
        normalize=args.normalize
    )

if __name__ == '__main__':
    main()
