#!/usr/bin/env python3
# File: predict_attention.py

import os
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split
from scipy.stats import pearsonr, spearmanr

# import your dataset and model definitions
from MLP_attention_scores_new import AttentionDataset, collate_fn, DDGPredictorAttention

def evaluate_loader(model, loader, device):
    ys, ps = [], []
    model.eval()
    with torch.no_grad():
        for xf, xr, sc, mask, y in loader:
            xf, xr, sc, mask = [t.to(device) for t in (xf, xr, sc, mask)]
            pred = model(xf, xr, sc, mask).cpu().numpy()
            ys.extend(y.numpy())
            ps.extend(pred)
    pear = pearsonr(ys, ps)[0]
    spear = spearmanr(ys, ps)[0]
    return pear, spear


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv',        default='data/SKEMPI2/SKEMPI2.csv')
    parser.add_argument('--wt_dir',     default='data/SKEMPI2/SKEMPI2_cache/embedding_wildtype_full_2048')
    parser.add_argument('--mut_dir',    default='data/SKEMPI2/SKEMPI2_cache/embedding_optimized_full_2048')
    parser.add_argument('--scores_dir', default='data/scores_cache')
    parser.add_argument('--best',       action='store_true', help='Use best_model.pt instead of final_model.pt')
    parser.add_argument('--model_best', default='best_model.pt')
    parser.add_argument('--model_final',default='final_model.pt')
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--test_size',  type=float, default=0.20)
    parser.add_argument('--val_size',   type=float, default=0.10)
    parser.add_argument('--num_workers',type=int, default=1)
    parser.add_argument('--pin_memory', action='store_true')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)

    # load dataset
    ds = AttentionDataset(args.csv, args.wt_dir, args.mut_dir, args.scores_dir)
    # split into train, val, test by unique cid
    unique_ids = np.unique(ds.ids)
    train_ids, test_ids = train_test_split(unique_ids,
                                           test_size=args.test_size,
                                           random_state=42)
    train_mask = np.isin(ds.ids, train_ids)
    train_idx_full = np.where(train_mask)[0]
    test_idx = np.where(~train_mask)[0]
    # from train split off validation
    train_idx, val_idx = train_test_split(train_idx_full,
                                           test_size=args.val_size,
                                           random_state=42)

    ds_val = Subset(ds, val_idx)
    ds_te  = Subset(ds, test_idx)

    loader_val = DataLoader(
        ds_val,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory
    )
    loader_te = DataLoader(
        ds_te,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory
    )

    # instantiate model
    model = DDGPredictorAttention(
        emb_dim=768,
        proj_dim=128,
        n_heads=2,
        n_layers=1,
        dropout=0.2
    ).to(device)

    # load checkpoint
    ckpt = args.model_best if args.best else args.model_final
    if not os.path.exists(ckpt):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt}")
    state = torch.load(ckpt, map_location=device)
    model.load_state_dict(state)

    # evaluate on validation set
    pear_val, spear_val = evaluate_loader(model, loader_val, device)
    print(f"Validation set results -> Pearson: {pear_val:.4f}, Spearman: {spear_val:.4f}")

    # evaluate on test set
    pear_test, spear_test = evaluate_loader(model, loader_te, device)
    print(f"Test set results       -> Pearson: {pear_test:.4f}, Spearman: {spear_test:.4f}")

if __name__ == '__main__':
    main()
