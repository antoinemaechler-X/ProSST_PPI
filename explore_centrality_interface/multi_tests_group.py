#!/usr/bin/env python3
import os
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from scipy.stats import pearsonr, spearmanr, f_oneway
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.metrics import r2_score
from tqdm import tqdm
import statsmodels.api as sm

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
                              normalize=False, except_high=False):
    """
    Run regression on combined data from the first N proteins using group-based CV
    and plot real vs. predicted, with R², Pearson and Spearman shown in a box.
    Only single-point mutations are included.
    If normalize=True, scale ddG to [0,1].
    If except_high=True, exclude mutations with ddG > 5.0.
    """
    proteins = get_first_N_proteins(csv_path, N)
    X1, X2, y_list, lengths, prot_ids = [], [], [], [], []
    mutation_info = []  # Store mutation details for high ddG reporting

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
            
            # Skip high ddG mutations if except_high is True
            if except_high and ddg > 5.0:
                continue

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
            prot_ids.append(p)
            
            # Store mutation info for high ddG reporting
            mutation_info.append({
                'protein': p,
                'mutation': muts[0],
                'ddG': ddg,
                'burial': b,
                'interface': i
            })

    # convert to arrays
    X = np.column_stack([X1, X2])
    y = np.array(y_list)
    lengths = np.array(lengths)
    prot_ids = np.array(prot_ids)

    # Filter out proteins with fewer than 3 mutations
    counts = pd.Series(prot_ids).value_counts()
    valid_prots = counts[counts >= 3].index.tolist()
    mask = np.isin(prot_ids, valid_prots)
    X, y, prot_ids = X[mask], y[mask], prot_ids[mask]
    proteins = valid_prots
    print(f"Using {len(proteins)} proteins with at least 3 mutations")

    # normalize ddG if requested
    if normalize:
        y_min, y_max = y.min(), y.max()
        if y_max > y_min:
            y = (y - y_min) / (y_max - y_min)
        else:
            y = np.zeros_like(y)

    # build pipeline: identity or polynomial
    if degree == 2:
        poly = PolynomialFeatures(degree=2, include_bias=False)
        X_poly = poly.fit_transform(X)
        feature_names = poly.get_feature_names_out(['burial', 'interface'])
        estimator = Pipeline([
            ('poly', poly),
            ('lr', LinearRegression())
        ])
    elif degree == 3:
        poly = PolynomialFeatures(degree=3, include_bias=False)
        X_poly = poly.fit_transform(X)
        feature_names = poly.get_feature_names_out(['burial', 'interface'])
        estimator = Pipeline([
            ('poly', poly),
            ('lr', LinearRegression())
        ])
    else:
        X_poly = X
        feature_names = ['burial', 'interface']
        estimator = Pipeline([
            ('id', FunctionTransformer(lambda x: x, validate=False)),
            ('lr', LinearRegression())
        ])

    # Fit the model to get p-values
    model = sm.OLS(y, sm.add_constant(X_poly)).fit()
    p_values = model.pvalues[1:]  # Skip the constant term

    # Group-based cross-validation
    print("Running group-based cross-validation...")
    n_splits = min(10, len(np.unique(prot_ids)))
    group_kfold = GroupKFold(n_splits=n_splits)
    y_pred = cross_val_predict(estimator, X, y, groups=prot_ids, cv=group_kfold)

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
        f'{N} proteins | {n_splits}-fold GroupCV | mode={mode}, k={neighbors}, '
        f'σ={sigma_interface}, deg={degree}' +
        (' | normalized' if normalize else '') +
        (' | except_high' if except_high else '')
    )

    # Create metrics text with p-values
    metrics = f"R²={r2:.2f}\nPearson={pr:.2f}\nSpearman={sr:.2f}\n\np-values:"
    for name, pval in zip(feature_names, p_values):
        metrics += f"\n{name}: {pval:.2e}"

    ax.text(0.05, 0.95, metrics, transform=ax.transAxes,
            fontsize=8, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))

    # save plot
    out_dir = os.path.join(os.path.dirname(__file__), 'plots_centrality_interface')
    os.makedirs(out_dir, exist_ok=True)
    fn = (f'GroupCV_{N}_prot_{n_splits}fold_deg{degree}_k{neighbors}'
          f'_s{sigma_interface}' + 
          ('_norm' if normalize else '') +
          ('_except_high' if except_high else '') + '.png')
    out_path = os.path.join(out_dir, fn)
    plt.tight_layout()
    plt.savefig(out_path)
    print(f"Saved {out_path}")

    # Print high ddG mutations
    high_ddg = [m for m in mutation_info if abs(m['ddG']) > 5.0]
    if high_ddg:
        print("\nMutations with |ddG| > 5.0:")
        print("Protein\tMutation\t|ddG|\tBurial\tInterface")
        print("-" * 60)
        for m in sorted(high_ddg, key=lambda x: abs(x['ddG']), reverse=True):
            print(f"{m['protein']}\t{m['mutation']}\t{abs(m['ddG']):.2f}\t{m['burial']:.3f}\t{m['interface']:.3f}")


def main():
    parser = argparse.ArgumentParser(
        description="Run regression across multiple proteins with group-based CV"
    )
    parser.add_argument('-N', '--num_proteins', type=int, default=40,
                        help='Number of proteins to include (default: 40)')
    parser.add_argument('-m', '--mode',
                        choices=['mean', 'sum', 'max'], default='mean',
                        help="Aggregation mode (default: mean)")
    parser.add_argument('-k', '--neighbors', type=int, default=9,
                        help='k for burial score (default: 9)')
    parser.add_argument('-s', '--sigma-interface', dest='sigma_interface',
                        type=float, default=2.5,
                        help='σ for interface Gaussian (default: 2.5)')
    parser.add_argument('-d', '--degree', type=int, choices=[1, 2, 3],
                        default=2, help='Regression degree (1 or 2)')
    parser.add_argument('--distinguish', action='store_true', default=False,
                        help='Color‐code points by number of mutations (no effect, single only)')
    parser.add_argument('--normalize', action='store_true', default=False,
                        help='Normalize ddG to [0,1] before regression')
    parser.add_argument('--except_high', action='store_true', default=False,
                        help='Exclude mutations with ddG > 5.0')
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
        normalize=args.normalize,
        except_high=args.except_high
    )

if __name__ == '__main__':
    main() 