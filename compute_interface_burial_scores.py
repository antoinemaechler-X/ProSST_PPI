#!/usr/bin/env python3
# File: compute_scores.py

import os
import argparse
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from explore_centrality_interface.burial_score import _get_cb_coordinates, _compute_score_for_mutation
from explore_centrality_interface.interface_score import _get_cb_coordinates as _get_cb_i, _compute_interface_score_for_mutation


def compute_and_save_scores(csv_path, pdb_dir, out_dir,
                            neighbor_count=9, sigma_interface=2.5,
                            mode='burial_interface'):
    """
    For each entry in the CSV, compute per-residue burial and interface scores
    and save to a .npz file in out_dir, named by PDB key.
    """
    os.makedirs(out_dir, exist_ok=True)
    df = pd.read_csv(csv_path).dropna(subset=['ddG']).reset_index(drop=True)

    for _, row in tqdm(df.iterrows(), total=len(df), desc='Compute scores'):
        key = row['#Pdb']  # e.g. '0_1CSE'
        pdb_file = os.path.join(pdb_dir, f"{key}.pdb")
        if not os.path.exists(pdb_file):
            tqdm.write(f"Warning: {pdb_file} not found, skipping")
            continue

        # burial
        cb_coords, keys_b, coords_b = _get_cb_coordinates(pdb_file)
        burial = []
        for chain, res, _ in keys_b:
            mut = f"A{chain}{res}A"
            try:
                score = _compute_score_for_mutation(
                    cb_coords, keys_b, coords_b, mut, neighbor_count)
            except Exception:
                score = 0.0
            burial.append(score)
        burial = np.array(burial, dtype=float)

        # interface
        cb_coords_i, keys_i, coords_i = _get_cb_i(pdb_file)
        interface = []
        for chain, res, _ in keys_i:
            mut = f"A{chain}{res}A"
            try:
                score = _compute_interface_score_for_mutation(
                    cb_coords_i, keys_i, coords_i, mut, sigma_interface)
            except Exception:
                score = 0.0
            interface.append(score)
        interface = np.array(interface, dtype=float)

        # normalize both to [0,1]
        if burial.max() > 0:
            burial /= burial.max()
        if interface.max() > 0:
            interface /= interface.max()

        # save
        out_path = os.path.join(out_dir, f"scores_{key}.npz")
        np.savez(out_path,
                 keys=np.array(keys_b, dtype=object),
                 burial=burial,
                 interface=interface)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Precompute burial and interface scores per residue')
    parser.add_argument('--csv', default='data/SKEMPI2/SKEMPI2.csv',
                        help='SKEMPI2 mutations CSV')
    parser.add_argument('--pdb_dir', default='data/SKEMPI2/SKEMPI2_cache/optimized',
                        help='Directory containing PDB files (optimized or wildtype)')
    parser.add_argument('--out_dir', default='data/scores_cache',
                        help='Output directory for per-PDB .npz score files')
    parser.add_argument('--neighbor_count', type=int, default=9,
                        help='Neighbor count for burial score')
    parser.add_argument('--sigma_interface', type=float, default=2.5,
                        help='Sigma for interface score')
    args = parser.parse_args()

    # Compute for optimized and wildtype separately
    compute_and_save_scores(args.csv, args.pdb_dir, args.out_dir,
                            neighbor_count=args.neighbor_count,
                            sigma_interface=args.sigma_interface)
