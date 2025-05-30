#!/usr/bin/env python3
import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
from scipy import stats

from burial_score import burial_score, burial_scores
from interface_score import interface_score, interface_scores
from dihedral_score import compute_dihedral_score, compute_dihedral_scores
from flexibility_score import flexibility_score, flexibility_scores

def main():
    parser = argparse.ArgumentParser(
        description="Analyze ddG vs normalized burial, interface, dihedral & flexibility scores"
    )
    parser.add_argument("-p", "--protein", type=str, required=True,
                        help="PDB code to filter (e.g. 1A22)")
    parser.add_argument("-n", "--num_lines", type=int, default=None,
                        help="Rows to process after filtering")
    parser.add_argument("-m", "--mode",
                        choices=["mean", "max", "individual", "sum"],
                        default="mean",
                        help="Aggregation mode")
    parser.add_argument("-k", "--neighbors", type=int, default=9,
                        help="k for burial (default: 9)")
    parser.add_argument("-s", "--sigma-interface", type=float,
                        dest="sigma_interface", default=1.0,
                        help="σ for interface Gaussian (default: 1.0)")
    parser.add_argument("--w_dihedral", type=float, default=1.0,
                        help="Weight for dihedral score")
    parser.add_argument("--min_flex_change", type=float, default=1e-5,
                        help="Minimum flexibility change to consider (default: 1e-5)")
    parser.add_argument("--single", action="store_true", default=False,
                        help="Only consider single-point mutations")
    parser.add_argument("--names", action="store_true", default=False,
                        help="Show mutation names on the plots")
    parser.add_argument("--distinguish", action="store_true", default=False,
                        help="Color‐code points by number of mutations (1,2,3,...)")
    args = parser.parse_args()

    here = os.path.dirname(os.path.abspath(__file__))
    plots_dir = os.path.join(here, 'plots_centrality_interface_dihedral_flex')
    os.makedirs(plots_dir, exist_ok=True)

    csv_path = os.path.join(here, '..', 'data', 'SKEMPI2', 'SKEMPI2.csv')
    pdb_wt_dir = os.path.join(here, '..', 'data', 'SKEMPI2',
                              'SKEMPI2_cache', 'wildtype')
    pdb_opt_dir = os.path.join(here, '..', 'data', 'SKEMPI2',
                               'SKEMPI2_cache', 'optimized')
    flex_wt_file = os.path.join(here, '..', 'data', 'SKEMPI2',
                                'SKEMPI2_cache', 'skempi_output_wildtype-3D-all-predictions.txt')
    flex_mt_file = os.path.join(here, '..', 'data', 'SKEMPI2',
                                'SKEMPI2_cache', 'skempi_output-3D-all-predictions.txt')

    df = pd.read_csv(csv_path)
    df = df[df['#Pdb_origin'] == args.protein]

    # If requested, keep only rows with exactly one mutation
    if args.single:
        single_mask = (
            df['Mutation(s)_cleaned']
              .str.strip('"')
              .str.split(',')
              .str.len() == 1
        )
        df = df[single_mask]

    if args.num_lines is not None:
        df = df.iloc[:args.num_lines]

    ddG_list        = []
    burial_list     = []
    interface_list  = []
    dihedral_list   = []
    flex_list       = []
    flex_ddG_list   = []  # List for signed ddG values (only for flexibility)
    labels          = []
    lengths         = []

    # wrap outer loop with tqdm
    for idx, row in tqdm(df.iterrows(),
                         total=len(df),
                         desc="Processing rows"):
        wt_pdb  = os.path.join(pdb_wt_dir,  f"{row['#Pdb']}.pdb")
        opt_pdb = os.path.join(pdb_opt_dir, f"{row['#Pdb']}.pdb")

        muts = row['Mutation(s)_cleaned'].strip('"').split(',')
        n_muts = len(muts)

        try:
            ddg_val = abs(float(row['ddG']))  # Keep absolute for burial/interface/dihedral
            ddg_signed = float(row['ddG'])    # Keep signed for flexibility
        except:
            ddg_val = float('nan')
            ddg_signed = float('nan')

        if args.mode == 'individual':
            # nested tqdm for mutations
            for mut in tqdm(muts,
                            desc=f"  muts in row {idx}",
                            leave=False):
                try:
                    b = burial_score(wt_pdb, mut, neighbor_count=args.neighbors)
                except Exception as e:
                    print(f"[Warn] burial row {idx}, mut {mut}: {e}")
                    b = float('nan')
                try:
                    i = interface_score(wt_pdb, mut,
                                        sigma_interface=args.sigma_interface)
                except Exception as e:
                    print(f"[Warn] interface row {idx}, mut {mut}: {e}")
                    i = float('nan')
                try:
                    d = compute_dihedral_score(wt_pdb, opt_pdb, mut)
                except Exception as e:
                    print(f"[Warn] dihedral row {idx}, mut {mut}: {e}")
                    d = float('nan')
                try:
                    f = flexibility_score(
                        flex_wt_file, flex_mt_file,
                        f"{row['#Pdb']}_{mut}",
                        min_change_threshold=args.min_flex_change
                    )
                except Exception as e:
                    print(f"[Warn] flexibility row {idx}, mut {mut}: {e}")
                    f = float('nan')

                burial_list.append(b)
                interface_list.append(i)
                dihedral_list.append(d)
                flex_list.append(f)
                ddG_list.append(ddg_val)  # Use absolute for burial/interface/dihedral
                flex_ddG_list.append(ddg_signed)  # Use signed only for flexibility
                labels.append(mut)
                lengths.append(n_muts)

        else:
            try:
                b_scores = burial_scores(wt_pdb, muts, neighbor_count=args.neighbors)
                i_scores = interface_scores(wt_pdb, muts, sigma_interface=args.sigma_interface)
                d_scores = compute_dihedral_scores(wt_pdb, opt_pdb, muts)
                f_scores = flexibility_scores(
                    flex_wt_file, flex_mt_file,
                    [f"{row['#Pdb']}_{mut}" for mut in muts],
                    min_change_threshold=args.min_flex_change
                )
                
                if args.mode == 'mean':
                    b = sum(b_scores) / len(b_scores)
                    i = sum(i_scores) / len(i_scores)
                    d = sum(d_scores) / len(d_scores)
                    f = sum(f_scores) / len(f_scores)
                elif args.mode == 'sum':
                    b = sum(b_scores)
                    i = sum(i_scores)
                    d = sum(d_scores)
                    f = sum(f_scores)
                else:  # max
                    b = max(b_scores)
                    i = max(i_scores)
                    d = max(d_scores)
                    f = max(f_scores)
            except Exception as e:
                print(f"[Warn] row {idx} aggregate: {e}")
                b, i, d, f = float('nan'), float('nan'), float('nan'), float('nan')

            burial_list.append(b)
            interface_list.append(i)
            dihedral_list.append(d)
            flex_list.append(f)
            ddG_list.append(ddg_val)  # Use absolute for burial/interface/dihedral
            flex_ddG_list.append(ddg_signed)  # Use signed only for flexibility
            labels.append(";".join(muts))
            lengths.append(n_muts)

    def make_scatter(x, y, labels, lengths, xlabel, ylabel, title, fn):
        plt.figure()
        if args.distinguish:
            unique_counts = sorted(set(lengths))
            cmap = plt.cm.get_cmap('tab10', len(unique_counts))
            color_map = {cnt: cmap(i) for i, cnt in enumerate(unique_counts)}
            colors = [color_map[cnt] for cnt in lengths]
            for cnt in unique_counts:
                plt.scatter([], [], c=[color_map[cnt]], label=f"{cnt} muts")
            plt.legend(title="Num mutations")
            plt.scatter(x, y, c=colors, alpha=0.7)
        else:
            plt.scatter(x, y, alpha=0.7)

        if args.names:
            for xi, yi, lab in zip(x, y, labels):
                plt.text(xi, yi, lab, fontsize=6, alpha=0.7)

        # Calculate correlation statistics
        # Remove any NaN values for correlation calculation
        mask = ~(np.isnan(x) | np.isnan(y))
        x_clean = np.array(x)[mask]
        y_clean = np.array(y)[mask]
        
        if len(x_clean) > 1:  # Only calculate if we have enough valid points
            pearson_r, pearson_p = stats.pearsonr(x_clean, y_clean)
            spearman_r, spearman_p = stats.spearmanr(x_clean, y_clean)
            
            # Create text box with correlation statistics
            stats_text = f'Pearson: r={pearson_r:.3f} (p={pearson_p:.2e})\nSpearman: ρ={spearman_r:.3f} (p={spearman_p:.2e})'
            plt.text(0.05, 0.95, stats_text,
                    transform=plt.gca().transAxes,
                    bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray'),
                    verticalalignment='top',
                    fontsize=8)

        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(title)
        plt.tight_layout()
        out = os.path.join(plots_dir, fn)
        plt.savefig(out)
        print(f"Saved {out}")

    mode_label = args.mode.capitalize()

    # All plots except flexibility use absolute ddG values
    make_scatter(
        burial_list, ddG_list, labels, lengths,
        'Normalized Burial', 'Absolute ddG',
        f'ddG vs Burial ({mode_label}, k={args.neighbors})',
        f'ddG_vs_burial_{args.protein}_{args.mode}_k{args.neighbors}.png'
    )
    make_scatter(
        interface_list, ddG_list, labels, lengths,
        'Normalized Interface', 'Absolute ddG',
        f'ddG vs Interface ({mode_label}, σ={args.sigma_interface})',
        f'ddG_vs_interface_{args.protein}_{args.mode}_s{args.sigma_interface}.png'
    )
    make_scatter(
        dihedral_list, ddG_list, labels, lengths,
        'Dihedral Score', 'Absolute ddG',
        f'ddG vs Dihedral Score ({mode_label})',
        f'ddG_vs_dihedral_{args.protein}_{args.mode}.png'
    )
    # Only flexibility plot uses signed ddG values
    make_scatter(
        flex_list, flex_ddG_list, labels, lengths,
        'Flexibility Score', 'Signed ddG',
        f'ddG vs Flexibility ({mode_label}, min_change={args.min_flex_change})',
        f'ddG_vs_flex_{args.protein}_{args.mode}_min{args.min_flex_change}.png'
    )

if __name__ == "__main__":
    main()
