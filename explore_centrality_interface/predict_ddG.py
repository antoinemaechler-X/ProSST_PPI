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
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.metrics import r2_score
import statsmodels.api as sm

from burial_score import burial_score, burial_scores
from interface_score import interface_score, interface_scores

def load_data(csv_path, pdb_dir, num_lines, mode,
              neighbor_count, sigma_interface, protein, except_high=False):
    df = pd.read_csv(csv_path)
    df = df[df['#Pdb_origin'] == protein]
    df = df[pd.to_numeric(df['ddG'], errors='coerce').notnull()]  # Drop invalid ddG

    if num_lines is not None:
        df = df.iloc[:num_lines]

    X1, X2, y, labels, lengths, mutation_info = [], [], [], [], [], []
    for _, row in df.iterrows():
        pdb_file = os.path.join(pdb_dir, f"{row['#Pdb']}.pdb")
        muts = row['Mutation(s)_cleaned'].strip('"').split(',')
        n_muts = len(muts)
        ddg_val = abs(float(row['ddG']))

        # Skip high ddG mutations if except_high is True
        if except_high and ddg_val > 5.0:
            continue

        if mode == 'individual':
            for mut in muts:
                b = burial_score(pdb_file, mut, neighbor_count=neighbor_count)
                i = interface_score(pdb_file, mut, sigma_interface=sigma_interface)
                X1.append(b); X2.append(i); y.append(ddg_val)
                labels.append(mut)
                lengths.append(n_muts)
                mutation_info.append({
                    'protein': protein,
                    'mutation': mut,
                    'ddG': ddg_val,
                    'burial': b,
                    'interface': i
                })
        else:
            b_list = burial_scores(pdb_file, muts, neighbor_count=neighbor_count)
            i_list = interface_scores(pdb_file, muts, sigma_interface=sigma_interface)
            if not b_list or not i_list:
                continue
            if mode == 'mean':
                b, i = np.mean(b_list), np.mean(i_list)
            elif mode == 'sum':
                b, i = np.sum(b_list), np.sum(i_list)
            else:  # max
                b, i = np.max(b_list), np.max(i_list)
            X1.append(b); X2.append(i); y.append(ddg_val)
            labels.append(";".join(muts))
            lengths.append(n_muts)
            mutation_info.append({
                'protein': protein,
                'mutation': ";".join(muts),
                'ddG': ddg_val,
                'burial': b,
                'interface': i
            })

    X = np.column_stack([X1, X2])
    return X, np.array(y), labels, lengths, mutation_info

def main():
    parser = argparse.ArgumentParser(
        description="Predict ddG via regression with 10-fold CV"
    )
    parser.add_argument('-p', '--protein', required=True,
                        help='PDB code (e.g. 1A22)')
    parser.add_argument('-n', '--num_lines', type=int, default=None,
                        help='Rows after filtering')
    parser.add_argument('-m', '--mode',
                        choices=['mean','max','individual','sum'],
                        default='mean')
    parser.add_argument('-k', '--neighbors', type=int, default=9,
                        help='k for burial (default:9)')
    parser.add_argument('-s', '--sigma-interface', type=float,
                        dest='sigma_interface', default=2.5,
                        help='σ for interface (default:2.5)')
    parser.add_argument('-d', '--degree', type=int, choices=[1,2],
                        default=1,
                        help='Degree of regression: 1=linear, 2=poly')
    parser.add_argument('--distinguish', action='store_true', default=False,
                        help='Color-code points by number of mutations')
    parser.add_argument('--single', action='store_true', default=False,
                        help='Only include single-mutation data')
    parser.add_argument('--except_high', action='store_true', default=False,
                        help='Exclude mutations with ddG > 5.0')
    args = parser.parse_args()

    here = os.path.dirname(os.path.abspath(__file__))
    plots_dir = os.path.join(here, 'plots_centrality_interface')
    os.makedirs(plots_dir, exist_ok=True)
    csv_path = os.path.join(here, '..', 'data', 'SKEMPI2', 'SKEMPI2.csv')
    pdb_dir  = os.path.join(here, '..', 'data', 'SKEMPI2', 'SKEMPI2_cache', 'wildtype')

    X, y, labels, lengths, mutation_info = load_data(
        csv_path, pdb_dir, args.num_lines,
        args.mode, args.neighbors,
        args.sigma_interface, args.protein,
        except_high=args.except_high
    )

    # optional single-mutation filter
    if args.single:
        mask = [l == 1 for l in lengths]
        X = X[mask]
        y = y[mask]
        labels = [lab for lab, m in zip(labels, mask) if m]
        lengths = [l for l in lengths if l == 1]
        mutation_info = [m for m, mask_val in zip(mutation_info, mask) if mask_val]

    # build pipeline: identity or polynomial
    if args.degree == 2:
        poly = PolynomialFeatures(degree=2, include_bias=False)
        X_poly = poly.fit_transform(X)
        feature_names = poly.get_feature_names_out(['burial', 'interface'])
        pipeline = Pipeline([
            ('poly', poly),
            ('lr', LinearRegression())
        ])
    else:
        X_poly = X
        feature_names = ['burial', 'interface']
        pipeline = Pipeline([
            ('id', FunctionTransformer(lambda x: x, validate=False)),
            ('lr', LinearRegression())
        ])

    # Fit the model to get p-values
    model = sm.OLS(y, sm.add_constant(X_poly)).fit()
    p_values = model.pvalues[1:]  # Skip the constant term

    # 10-fold cross-validation
    cv = KFold(n_splits=10, shuffle=True, random_state=42)
    y_pred = cross_val_predict(pipeline, X, y, cv=cv)

    # compute metrics on all folds
    r2 = r2_score(y, y_pred)
    pr, _ = pearsonr(y, y_pred)
    sr, _ = spearmanr(y, y_pred)

    # plot true vs predicted across all folds
    fig, ax = plt.subplots()
    if args.distinguish and not args.single:
        counts = sorted(set(lengths))
        cmap = plt.cm.get_cmap('tab10', len(counts))
        color_map = {cnt: cmap(i) for i, cnt in enumerate(counts)}
        colors = [color_map[cnt] for cnt in lengths]
        for cnt in counts:
            ax.scatter([], [], c=[color_map[cnt]], label=f"{cnt} muts")
        ax.legend(title="# mutations")
        ax.scatter(y, y_pred, c=colors, alpha=0.7)
    else:
        ax.scatter(y, y_pred, alpha=0.7)

    mn, mx = min(y.min(), y_pred.min()), max(y.max(), y_pred.max())
    ax.plot([mn, mx], [mn, mx], 'k--', lw=1)
    ax.set_xlabel('True Absolute ddG')
    ax.set_ylabel('Predicted Absolute ddG')
    ax.set_title(
        f"{args.protein} | 10-fold CV | deg={args.degree}, mode={args.mode}, "
        f"k={args.neighbors}, σ={args.sigma_interface}" +
        (' | except_high' if args.except_high else '')
    )

    # Create metrics text with p-values
    metrics = f"R²={r2:.2f}\nPearson={pr:.2f}\nSpearman={sr:.2f}\n\np-values:"
    for name, pval in zip(feature_names, p_values):
        metrics += f"\n{name}: {pval:.2e}"

    ax.text(0.05, 0.95, metrics, transform=ax.transAxes,
            fontsize=8, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))

    # save
    out_fn = (
        f"ddG_cv10_deg{args.degree}_{args.protein}_{args.mode}"
        f"{'_single' if args.single else ''}"
        f"_k{args.neighbors}_s{args.sigma_interface}"
        f"{'_except_high' if args.except_high else ''}.png"
    )
    out_path = os.path.join(plots_dir, out_fn)
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

if __name__ == '__main__':
    main()
