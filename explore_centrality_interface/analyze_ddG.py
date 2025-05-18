#!/usr/bin/env python3
import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

from burial_score import burial_score, burial_scores
from interface_score import interface_score, interface_scores
from shift_score import combined_shift_score, combined_shift_scores

def main():
    parser = argparse.ArgumentParser(
        description="Analyze ddG vs normalized burial, interface & shift scores"
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
    parser.add_argument("--w_dihedral_shift", type=float, default=1.0,
                        help="Weight for dihedral component in shift score")
    parser.add_argument("--w_contact_shift", type=float, default=1.0,
                        help="Weight for contact component in shift score")
    parser.add_argument("--single", action="store_true", default=False,
                        help="Only consider single-point mutations")
    parser.add_argument("--names", action="store_true", default=False,
                        help="Show mutation names on the plots")
    parser.add_argument("--distinguish", action="store_true", default=False,
                        help="Color‐code points by number of mutations (1,2,3,...)")
    args = parser.parse_args()

    here = os.path.dirname(os.path.abspath(__file__))
    plots_dir = os.path.join(here, 'plots_centrality_interface_shift')
    os.makedirs(plots_dir, exist_ok=True)

    csv_path = os.path.join(here, '..', 'data', 'SKEMPI2', 'SKEMPI2.csv')
    pdb_wt_dir = os.path.join(here, '..', 'data', 'SKEMPI2',
                              'SKEMPI2_cache', 'wildtype')
    pdb_opt_dir = os.path.join(here, '..', 'data', 'SKEMPI2',
                               'SKEMPI2_cache', 'optimized')

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
    shift_list      = []
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
            ddg_val = abs(float(row['ddG']))
        except:
            ddg_val = float('nan')

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
                    s = combined_shift_score(
                        wt_pdb, opt_pdb, mut,
                        w_dihedral=args.w_dihedral_shift,
                        w_contact=args.w_contact_shift
                    )
                except Exception as e:
                    print(f"[Warn] shift row {idx}, mut {mut}: {e}")
                    s = float('nan')

                burial_list.append(b)
                interface_list.append(i)
                shift_list.append(s)
                ddG_list.append(ddg_val)
                labels.append(mut)
                lengths.append(n_muts)

        else:
            try:
                b_scores = burial_scores(wt_pdb, muts, neighbor_count=args.neighbors)
                i_scores = interface_scores(wt_pdb, muts, sigma_interface=args.sigma_interface)
                s_scores = combined_shift_scores(
                    wt_pdb, opt_pdb, muts,
                    w_dihedral=args.w_dihedral_shift,
                    w_contact=args.w_contact_shift
                )
                if args.mode == 'mean':
                    b = sum(b_scores) / len(b_scores)
                    i = sum(i_scores) / len(i_scores)
                    s = sum(s_scores) / len(s_scores)
                elif args.mode == 'sum':
                    b = sum(b_scores)
                    i = sum(i_scores)
                    s = sum(s_scores)
                else:  # max
                    b = max(b_scores)
                    i = max(i_scores)
                    s = max(s_scores)
            except Exception as e:
                print(f"[Warn] row {idx} aggregate: {e}")
                b, i, s = float('nan'), float('nan'), float('nan')

            burial_list.append(b)
            interface_list.append(i)
            shift_list.append(s)
            ddG_list.append(ddg_val)
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

        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(title)
        plt.tight_layout()
        out = os.path.join(plots_dir, fn)
        plt.savefig(out)
        print(f"Saved {out}")

    mode_label = args.mode.capitalize()

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
        shift_list, ddG_list, labels, lengths,
        'Shift Score', 'Absolute ddG',
        (f'ddG vs Shift Score '
         f'({mode_label}, w_dihedral={args.w_dihedral_shift}, '
         f'w_contact={args.w_contact_shift})'),
        (f'ddG_vs_shift_{args.protein}_{args.mode}_'
         f'wD{args.w_dihedral_shift}_wC{args.w_contact_shift}.png')
    )

if __name__ == '__main__':
    main()
