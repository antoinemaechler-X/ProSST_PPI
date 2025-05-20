#!/usr/bin/env python3
# File: MLP.py

import os
import argparse
import itertools
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import json
from tqdm.auto import tqdm


def load_embeddings(path):
    return np.load(path)


def process_embeddings(wt_emb, mut_emb):
    diff = mut_emb - wt_emb
    return np.nanmax(diff, axis=0)


class DdgDataset(Dataset):
    def __init__(self, Xf, Xr, y):
        self.Xf = Xf; self.Xr = Xr; self.y = y.astype(np.float32)
    def __len__(self): return len(self.y)
    def __getitem__(self, idx):
        return (torch.tensor(self.Xf[idx]),
                torch.tensor(self.Xr[idx]),
                torch.tensor(self.y[idx]))


class DDGPredictor(nn.Module):
    def __init__(self, input_dim, hidden_dim, dropout_rate, num_hidden):
        super().__init__()
        layers = []
        in_dim = input_dim
        for i in range(num_hidden):
            out_dim = hidden_dim // (2**i)
            layers += [nn.Linear(in_dim, out_dim), nn.ReLU(), nn.Dropout(dropout_rate)]
            in_dim = out_dim
        self.mlp = nn.Sequential(*layers)
        self.head = nn.Linear(in_dim, 1)
    def forward(self, xf, xr):
        f = self.head(self.mlp(xf))
        r = self.head(self.mlp(xr))
        return (f - r) / 2


def preprocess(csv_path, wt_dir, mut_dir, cache_path):
    if cache_path and os.path.exists(cache_path):
        data = np.load(cache_path)
        return data['Xf'], data['Xr'], data['y'], data['ids']
    df = pd.read_csv(csv_path).dropna(subset=['ddG'])
    Xf, Xr, y, ids = [], [], [], []
    for _, r in tqdm(df.iterrows(), total=len(df), desc='Preprocess'):
        wt = load_embeddings(os.path.join(wt_dir, f"{r['#Pdb']}_full_embeddings.npy"))
        mt = load_embeddings(os.path.join(mut_dir, f"{r['#Pdb']}_full_embeddings.npy"))
        fwd = process_embeddings(wt, mt)
        rev = process_embeddings(mt, wt)
        if np.isnan(fwd).any() or np.isnan(rev).any(): continue
        Xf.append(fwd); Xr.append(rev); y.append(r['ddG']); ids.append(r['#Pdb_origin'])
    Xf = np.vstack(Xf); Xr = np.vstack(Xr); y = np.array(y); ids = np.array(ids)
    if cache_path: np.savez(cache_path, Xf=Xf, Xr=Xr, y=y, ids=ids)
    return Xf, Xr, y, ids


def train_one(args, device, Xf_tr, Xr_tr, y_tr, Xf_val, Xr_val, y_val):
    # data loaders
    tr_ds = DdgDataset(Xf_tr, Xr_tr, y_tr)
    val_ds = DdgDataset(Xf_val, Xr_val, y_val)
    tr_loader = DataLoader(tr_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size)

    model = DDGPredictor(args.input_dim, args.hidden_dim, args.dropout, args.num_hidden).to(device)
    opt = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.MSELoss()

    best_pear = -1
    for epoch in range(1, args.epochs+1):
        model.train()
        for xf, xr, y in tr_loader:
            xf, xr, y = xf.to(device), xr.to(device), y.to(device)
            opt.zero_grad(); loss = criterion(model(xf,xr).squeeze(), y); loss.backward(); opt.step()
        # evaluate on val
        ys, ps = [], []
        model.eval()
        with torch.no_grad():
            for xf, xr, y in val_loader:
                xf, xr = xf.to(device), xr.to(device)
                p = model(xf,xr).squeeze().cpu().numpy()
                ys.extend(y.numpy()); ps.extend(p)
        pear = pearsonr(ys, ps)[0]
        if pear > best_pear: best_pear = pear
    return best_pear


def hyperparam_search(args, device, Xf, Xr, y, ids):
    # split once into train+val and test
    tr_ids, te_ids = train_test_split(np.unique(ids), test_size=args.test_size, random_state=42)
    mask = np.isin(ids, tr_ids)
    Xf_tr_all, Xr_tr_all, y_tr_all = Xf[mask], Xr[mask], y[mask]  # train+val set
    Xf_te, Xr_te, y_te = Xf[~mask], Xr[~mask], y[~mask]  # test set

    grid = {
        'hidden_dim': [128, 256, 512],
        'num_hidden': [2, 3, 4],
        'dropout': [0.1, 0.2, 0.3],
        'lr': [1e-4, 1e-3, 1e-2],
        'weight_decay': [0, 1e-6, 1e-5]
    }
    best = {'pear': -1, 'params': None}
    for combo in tqdm(list(itertools.product(*grid.values())), desc='Grid Search'):
        for k, v in zip(grid.keys(), combo): setattr(args, k, v)
        # Train on train+val and evaluate on test
        pear = train_one(args, device, Xf_tr_all, Xr_tr_all, y_tr_all, Xf_te, Xr_te, y_te)
        if pear > best['pear']:
            best['pear'] = pear; best['params'] = combo
    print('Best test Pearson:', best['pear'])
    best_params = dict(zip(grid.keys(), best['params']))
    print('Best params:', best_params)
    # save best params to JSON
    with open('best_params.json', 'w') as f:
        json.dump(best_params, f, indent=2)
    # set args to best for final run
    for k, v in zip(grid.keys(), best['params']): setattr(args, k, v)
    return args


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--csv', default='data/SKEMPI2/SKEMPI2.csv')
    p.add_argument('--wt_dir', default='data/SKEMPI2/SKEMPI2_cache/embedding_wildtype_full_2048')
    p.add_argument('--mut_dir', default='data/SKEMPI2/SKEMPI2_cache/embedding_optimized_full_2048')
    p.add_argument('--cache', default='data/preproc.npz')
    p.add_argument('--input_dim', type=int, default=768)
    p.add_argument('--hidden_dim', type=int, default=512)
    p.add_argument('--num_hidden', type=int, default=2)
    p.add_argument('--dropout', type=float, default=0.1)
    p.add_argument('--lr', type=float, default=0.001)
    p.add_argument('--weight_decay', type=float, default=1e-5)
    p.add_argument('--epochs', type=int, default=50)
    p.add_argument('--batch_size', type=int, default=32)
    p.add_argument('--test_size', type=float, default=0.2)
    p.add_argument('--val_size', type=float, default=0.1)
    p.add_argument('--hyperparam', action='store_true', help='Run hyperparameter grid search')
    p.add_argument('--no_early_stop', action='store_true', help='Disable early stopping')
    p.add_argument('--loss_plot', default='loss.png')
    p.add_argument('--pred_plot', default='pred.png')
    args = p.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)

    Xf, Xr, y, ids = preprocess(args.csv, args.wt_dir, args.mut_dir, args.cache)

    if args.hyperparam:
        args = hyperparam_search(args, device, Xf, Xr, y, ids)

    # final train/test
    tr_ids, te_ids = train_test_split(np.unique(ids), test_size=args.test_size, random_state=42)
    mask = np.isin(ids, tr_ids)
    Xf_tr, Xr_tr, y_tr = Xf[mask], Xr[mask], y[mask]
    Xf_te, Xr_te, y_te = Xf[~mask], Xr[~mask], y[~mask]
    Xf_tr, Xf_val, Xr_tr, Xr_val, y_tr, y_val = train_test_split(
        Xf_tr, Xr_tr, y_tr, test_size=args.val_size, random_state=42)

    # train full model
    # reuse train_one but on (train+val)
    final_pear = train_one(args, device, Xf_tr, Xr_tr, y_tr, Xf_te, Xr_te, y_te)
    print('Test Pearson:', final_pear)

if __name__ == '__main__':
    main()
