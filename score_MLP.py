#!/usr/bin/env python3
# File: score_MLP.py

import os
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from scipy.stats import pearsonr, spearmanr
import matplotlib.pyplot as plt
from tqdm.auto import tqdm

# Import burial and interface scoring functions
from explore_centrality_interface.burial_score import _get_cb_coordinates, _compute_score_for_mutation
from explore_centrality_interface.interface_score import _compute_interface_score_for_mutation


def all_burial_scores(pdb_path, neighbor_count=9):
    cb_coords, keys, coords = _get_cb_coordinates(pdb_path)
    # raw burial per residue
    raw = []
    for chain, res, _ in keys:
        mut_str = f"A{chain}{res}A"
        score = _compute_score_for_mutation(cb_coords, keys, coords, mut_str, neighbor_count)
        raw.append(score)
    raw = np.array(raw, dtype=float)
    max_raw = raw.max() if raw.size else 1.0
    if max_raw == 0: return np.zeros_like(raw)
    return raw / max_raw


def all_interface_scores(pdb_path, sigma_interface=2.5):
    cb_coords, keys, coords = _get_cb_coordinates(pdb_path)
    raw = []
    for chain, res, _ in keys:
        mut_str = f"A{chain}{res}A"
        score = _compute_interface_score_for_mutation(cb_coords, keys, coords, mut_str, sigma_interface)
        raw.append(score)
    raw = np.array(raw, dtype=float)
    max_raw = raw.max() if raw.size else 1.0
    if max_raw == 0: return np.zeros_like(raw)
    return raw / max_raw


class DdgDataset(Dataset):
    """Dataset for ddG prediction"""
    def __init__(self, features, labels):
        self.X = features
        self.y = labels.astype(np.float32)

    def __len__(self): return len(self.y)

    def __getitem__(self, idx):
        return torch.tensor(self.X[idx], dtype=torch.float32), torch.tensor(self.y[idx], dtype=torch.float32)


class MLP(nn.Module):
    """Simple feedforward regressor"""
    def __init__(self, input_dim, hidden_dims, dropout):
        super().__init__()
        layers = []
        dims = [input_dim] + hidden_dims
        for i in range(len(hidden_dims)):
            layers += [nn.Linear(dims[i], dims[i+1]), nn.ReLU(), nn.Dropout(dropout)]
        layers.append(nn.Linear(dims[-1], 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x): return self.net(x).squeeze(-1)


def preprocess(args):
    # Load or compute features
    if args.cache and os.path.exists(args.cache):
        data = np.load(args.cache, allow_pickle=True)
        return data['features'], data['labels']

    df = pd.read_csv(args.csv).dropna(subset=['ddG']).reset_index(drop=True)
    features, labels = [], []
    for _, row in tqdm(df.iterrows(), total=len(df), desc='Preprocess'):
        key = row['#Pdb']
        # load embeddings
        wt_emb = np.load(os.path.join(args.wt_emb_dir, f"{key}_full_embeddings.npy"))
        mt_emb = np.load(os.path.join(args.mt_emb_dir, f"{key}_full_embeddings.npy"))
        # load pdbs
        wt_pdb = os.path.join(args.wt_pdb_dir, f"{key}.pdb")
        mt_pdb = os.path.join(args.mt_pdb_dir, f"{key}.pdb")
        # compute scores
        bur_wt = all_burial_scores(wt_pdb, args.neighbor_count)
        int_wt = all_interface_scores(wt_pdb, args.sigma_interface)
        scale_wt = bur_wt + int_wt
        bur_mt = all_burial_scores(mt_pdb, args.neighbor_count)
        int_mt = all_interface_scores(mt_pdb, args.sigma_interface)
        scale_mt = bur_mt + int_mt
        # scale embeddings per residue
        wt_scaled = wt_emb * scale_wt[:, None]
        mt_scaled = mt_emb * scale_mt[:, None]
        # diff + max pool
        diff = mt_scaled - wt_scaled
        feat = np.nanmax(diff, axis=0)
        features.append(feat)
        labels.append(row['ddG'])

    features = np.vstack(features)
    labels = np.array(labels, dtype=np.float32)
    if args.cache:
        np.savez(args.cache, features=features, labels=labels)
    return features, labels


def train(args, device, features, labels):
    # split on complexes
    df = pd.read_csv(args.csv).dropna(subset=['ddG']).reset_index(drop=True)
    ids = df['#Pdb_origin'].values
    complexes = np.unique(ids)
    np.random.seed(args.seed)
    np.random.shuffle(complexes)
    split1 = int((1-args.test_size) * len(complexes))
    train_c = complexes[:split1]
    test_c = complexes[split1:]
    mask_tr = np.isin(ids, train_c)
    X_tr, y_tr = features[mask_tr], labels[mask_tr]
    X_te, y_te = features[~mask_tr], labels[~mask_tr]
    # further val split
    X_tr, X_val, y_tr, y_val = train_test_split(X_tr, y_tr, test_size=args.val_size, random_state=args.seed)

    # loaders
    tr_loader = DataLoader(DdgDataset(X_tr,y_tr), batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(DdgDataset(X_val,y_val), batch_size=args.batch_size)
    te_loader = DataLoader(DdgDataset(X_te,y_te), batch_size=args.batch_size)

    # model
    model = MLP(features.shape[1], args.hidden_dims, args.dropout).to(device)
    opt = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.MSELoss()

    # training
    best_spear = -1
    history = {'train_loss':[], 'val_loss':[], 'val_spear':[]}
    for epoch in range(1, args.epochs+1):
        model.train()
        train_loss = 0
        for Xb, yb in tr_loader:
            Xb, yb = Xb.to(device), yb.to(device)
            opt.zero_grad()
            pred = model(Xb)
            loss = criterion(pred, yb)
            loss.backward(); opt.step()
            train_loss += loss.item()*Xb.size(0)
        train_loss /= len(tr_loader.dataset)

        # val
        model.eval()
        val_loss=0; ys=[]; preds=[]
        with torch.no_grad():
            for Xb, yb in val_loader:
                Xb, yb = Xb.to(device), yb.to(device)
                p = model(Xb).cpu().numpy(); ys.extend(yb.cpu().numpy()); preds.extend(p)
                val_loss += criterion(torch.tensor(p), yb.cpu()).item()*Xb.size(0)
        val_loss /= len(val_loader.dataset)
        spear = spearmanr(ys, preds)[0]
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_spear'].append(spear)
        print(f"Epoch {epoch}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, val_spear={spear:.3f}")

    # test
    ys=[]; preds=[]
    model.eval()
    with torch.no_grad():
        for Xb, yb in te_loader:
            p = model(Xb.to(device)).cpu().numpy(); ys.extend(yb.numpy()); preds.extend(p)
    spear_test = spearmanr(ys, preds)[0]
    print(f"Test Spearman: {spear_test:.3f}")

    # plots
    plt.figure(); plt.plot(history['train_loss'], label='Train Loss'); plt.plot(history['val_loss'], label='Val Loss'); plt.legend(); plt.savefig(args.loss_plot)
    plt.figure(); plt.scatter(ys, preds, alpha=0.5); mn,mx=min(ys+preds),max(ys+preds); plt.plot([mn,mx],[mn,mx],'--'); plt.text(0.05,0.95,f'Spearman: {spear_test:.3f}',transform=plt.gca().transAxes,va='top'); plt.savefig(args.pred_plot)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--csv', default='data/SKEMPI2/SKEMPI2.csv')
    p.add_argument('--wt_emb_dir', default='data/SKEMPI2/SKEMPI2_cache/embedding_wildtype_full_2048')
    p.add_argument('--mt_emb_dir', default='data/SKEMPI2/SKEMPI2_cache/embedding_optimized_full_2048')
    p.add_argument('--wt_pdb_dir', default='data/SKEMPI2/SKEMPI2_cache/wildtype')
    p.add_argument('--mt_pdb_dir', default='data/SKEMPI2/SKEMPI2_cache/optimized')
    p.add_argument('--cache', default='data/score_features.npz')
    p.add_argument('--neighbor_count', type=int, default=9)
    p.add_argument('--sigma_interface', type=float, default=2.5)
    p.add_argument('--input_dim', type=int, default=768)
    p.add_argument('--hidden_dims', nargs='+', type=int, default=[512,256,128])
    p.add_argument('--dropout', type=float, default=0.2)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--epochs', type=int, default=50)
    p.add_argument('--batch_size', type=int, default=32)
    p.add_argument('--test_size', type=float, default=0.2)
    p.add_argument('--val_size', type=float, default=0.1)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--loss_plot', default='score_loss.png')
    p.add_argument('--pred_plot', default='score_pred.png')
    args = p.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)

    features, labels = preprocess(args)
    train(args, device, features, labels)

if __name__ == '__main__':
    main()
