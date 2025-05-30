#!/usr/bin/env python3
# File: MLP_scores.py

import os
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from scipy.stats import pearsonr, spearmanr
import matplotlib.pyplot as plt
from tqdm.auto import tqdm


class EarlyStopping:
    """
    Early stopping with moving average and minimum delta.
    """
    def __init__(self, patience=20, min_delta=1e-4, window_size=10):
        self.patience = patience
        self.min_delta = min_delta
        self.window_size = window_size
        self.best_avg = float('inf')
        self.counter = 0
        self.losses = []
        self.best_state = None

    def __call__(self, loss, model):
        self.losses.append(loss)
        if len(self.losses) > self.window_size:
            self.losses.pop(0)
        avg = sum(self.losses) / len(self.losses)
        if avg < self.best_avg - self.min_delta:
            self.best_avg = avg
            self.counter = 0
            # save best state CPU-side
            self.best_state = {k: v.cpu() for k, v in model.state_dict().items()}
            return False
        else:
            self.counter += 1
            return self.counter >= self.patience


def load_embeddings(path):
    return np.load(path)


class DdgDataset(Dataset):
    """
    Dataset returning forward & reverse pooled features and ddG.
    """
    def __init__(self, Xf, Xr, y):
        self.Xf = Xf
        self.Xr = Xr
        self.y = y.astype(np.float32)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.Xf[idx], dtype=torch.float32),
            torch.tensor(self.Xr[idx], dtype=torch.float32),
            torch.tensor(self.y[idx], dtype=torch.float32)
        )


class DDGPredictor(nn.Module):
    """
    MLP predicting antisymmetric ddG: (fwd - rev)/2.
    """
    def __init__(self, input_dim, hidden_dim, dropout_rate, num_hidden):
        super().__init__()
        layers = []
        in_dim = input_dim
        for i in range(num_hidden):
            out_dim = hidden_dim if i == 0 else hidden_dim // (2**i)
            layers += [nn.Linear(in_dim, out_dim), nn.ReLU(), nn.Dropout(dropout_rate)]
            in_dim = out_dim
        self.mlp = nn.Sequential(*layers)
        self.head = nn.Linear(in_dim, 1)

    def forward(self, xf, xr):
        f = self.head(self.mlp(xf))
        r = self.head(self.mlp(xr))
        return (f - r) / 2


def preprocess(csv_path, wt_dir, mut_dir, scores_dir, cache_path=None):
    """
    Loads embeddings and scores, builds two-channel pooled features.
    """
    if cache_path and os.path.exists(cache_path):
        data = np.load(cache_path)
        return data['Xf'], data['Xr'], data['y'], data['ids']

    # load buried/interface scores
    scores_dict = {}
    for fname in os.listdir(scores_dir):
        if not fname.endswith('.npz'):
            continue
        mut_id = fname[len("scores_"):-4]
        arr = np.load(os.path.join(scores_dir, fname))
        scores_dict[mut_id] = {
            'burial': arr['burial'],
            'interface': arr['interface']
        }

    df = pd.read_csv(csv_path).dropna(subset=['ddG'])
    Xf, Xr, y, ids = [], [], [], []

    for _, row in tqdm(df.iterrows(), total=len(df), desc='Preprocess'):
        mut_id = row['#Pdb']
        cid = row['#Pdb_origin']
        try:
            wt = load_embeddings(os.path.join(wt_dir,  f"{mut_id}_full_embeddings.npy"))
            mt = load_embeddings(os.path.join(mut_dir, f"{mut_id}_full_embeddings.npy"))
        except FileNotFoundError:
            continue

        sc = scores_dict.get(mut_id)
        if sc is None:
            continue
        burial = sc['burial']
        interface = sc['interface']
        if wt.shape[0] != burial.shape[0] or mt.shape[0] != burial.shape[0]:
            continue

        diff_fwd = mt - wt
        diff_rev = wt - mt

        # two channels: interface-weighted maxpool, burial-weighted maxpool
        pi_fwd = np.max(diff_fwd * interface[:, None], axis=0)
        pi_rev = np.max(diff_rev * interface[:, None], axis=0)
        pb_fwd = np.max(diff_fwd * burial[:, None], axis=0)
        pb_rev = np.max(diff_rev * burial[:, None], axis=0)

        feat_fwd = np.concatenate([pi_fwd, pb_fwd], axis=0)
        feat_rev = np.concatenate([pi_rev, pb_rev], axis=0)

        Xf.append(feat_fwd)
        Xr.append(feat_rev)
        y.append(row['ddG'])
        ids.append(cid)

    if not Xf:
        raise ValueError("No data passed preprocessing! Check your inputs.")

    Xf = np.vstack(Xf)
    Xr = np.vstack(Xr)
    y = np.array(y, dtype=np.float32)
    ids = np.array(ids)
    if cache_path:
        np.savez(cache_path, Xf=Xf, Xr=Xr, y=y, ids=ids)
    return Xf, Xr, y, ids


def train_model(model, train_loader, test_loader, device,
                lr, epochs, weight_decay, patience,
                min_delta, window_size, lr_scheduler_patience,
                lr_factor, min_lr, corr_lambda=0.1):
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=lr_factor,
        patience=lr_scheduler_patience, verbose=True
    )
    stopper = EarlyStopping(patience, min_delta, window_size)

    best_pear = -np.inf
    train_losses, test_losses = [], []

    for epoch in range(1, epochs+1):
        model.train()
        run_tr = 0.0
        for xf, xr, y in tqdm(train_loader, desc=f'Epoch {epoch}', leave=False):
            xf, xr, y = xf.to(device), xr.to(device), y.to(device)

            optimizer.zero_grad()
            pred = model(xf, xr).squeeze()

            # unweighted MSE
            loss_mse = F.mse_loss(pred, y)
            # correlation loss (batch‐wise Pearson)
            pc = pred - pred.mean()
            yc = y    - y.mean()
            num = (pc * yc).mean()
            den = torch.sqrt((pc**2).mean() * (yc**2).mean()) + 1e-8
            pearson_batch = num / den
            loss_corr = 1 - pearson_batch

            loss = loss_mse + corr_lambda * loss_corr
            loss.backward()
            optimizer.step()

            run_tr += loss_mse.item() * xf.size(0)

        train_loss = run_tr / len(train_loader.dataset)
        train_losses.append(train_loss)

        # evaluate on test set each epoch
        model.eval()
        run_te = 0.0
        ys, ps = [], []
        with torch.no_grad():
            for xf, xr, y in test_loader:
                xf, xr, y = xf.to(device), xr.to(device), y.to(device)
                pred = model(xf, xr).squeeze()
                run_te += F.mse_loss(pred, y).item() * xf.size(0)
                ys.extend(y.cpu().numpy())
                ps.extend(pred.cpu().numpy())

        test_loss = run_te / len(test_loader.dataset)
        test_losses.append(test_loss)
        pear = pearsonr(ys, ps)[0]
        spear = spearmanr(ys, ps)[0]
        lr_cur = optimizer.param_groups[0]["lr"]
        print(f"Epoch {epoch}: Train={train_loss:.4f}, Test={test_loss:.4f}, "
              f"Pearson={pear:.3f}, Spearman={spear:.3f} | LR={lr_cur:.2e}")

        # save best by Pearson
        if pear > best_pear:
            best_pear = pear
            torch.save(model.state_dict(), 'best_model.pt')

        scheduler.step(test_loss)
        if lr_cur < min_lr:
            print(f"LR below {min_lr:.1e}, stopping.")
            break
        if stopper(test_loss, model):
            print("Early stopping!")
            model.load_state_dict(stopper.best_state)
            break

    torch.save(model.state_dict(), 'final_model.pt')
    return model, train_losses, test_losses


def plot_loss(train_losses, test_losses, out):
    plt.figure(figsize=(8,5))
    epochs = range(1, len(train_losses) + 1)
    plt.plot(epochs, train_losses, label='Train Loss')
    plt.plot(epochs, test_losses,  label='Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.title('Training and Test Loss')
    plt.legend()
    plt.tight_layout()
    plt.savefig(out)


def evaluate(model, loader, device, out_pred):
    ys, ps = [], []
    model.eval()
    with torch.no_grad():
        for xf, xr, y in loader:
            xf, xr = xf.to(device), xr.to(device)
            pred = model(xf, xr).squeeze().cpu().numpy()
            ys.extend(y.numpy())
            ps.extend(pred)

    pear = pearsonr(ys, ps)[0]
    spear= spearmanr(ys, ps)[0]

    plt.figure(figsize=(6,6))
    plt.scatter(ys, ps, alpha=0.6)
    mn, mx = min(ys+ps), max(ys+ps)
    plt.plot([mn, mx], [mn, mx], '--')
    plt.xlabel('True ddG')
    plt.ylabel('Predicted ddG')
    plt.title('True vs Pred')
    plt.text(0.05, 0.95, f'Pearson: {pear:.3f}\nSpearman: {spear:.3f}',
             transform=plt.gca().transAxes, va='top',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    plt.tight_layout()
    plt.savefig(out_pred)
    return pear, spear


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--csv',        default='data/SKEMPI2/SKEMPI2.csv')
    p.add_argument('--wt_dir',     default='data/SKEMPI2/SKEMPI2_cache/embedding_wildtype_full_2048')
    p.add_argument('--mut_dir',    default='data/SKEMPI2/SKEMPI2_cache/embedding_optimized_full_2048')
    p.add_argument('--scores_dir', default='data/scores_cache')
    p.add_argument('--cache',      default='data/preproc_with_scores.npz')
    p.add_argument('--input_dim',  type=int, default=768)
    p.add_argument('--hidden_dim', type=int, default=1024)
    p.add_argument('--num_hidden', type=int, default=5)
    p.add_argument('--dropout',    type=float, default=0.2)
    p.add_argument('--lr',         type=float, default=1e-5)
    p.add_argument('--epochs',     type=int,   default=150)
    p.add_argument('--batch_size', type=int,   default=32)
    p.add_argument('--test_size',  type=float, default=0.10)
    p.add_argument('--weight_decay', type=float, default=1e-4)
    p.add_argument('--patience',     type=int,   default=20)
    p.add_argument('--min_delta',    type=float, default=1e-5)
    p.add_argument('--win',          type=int,   default=10)
    p.add_argument('--lr_sched_pat', type=int,   default=5)
    p.add_argument('--lr_factor',    type=float, default=0.5)
    p.add_argument('--min_lr',       type=float, default=1e-6)
    p.add_argument('--corr_lambda',  type=float, default=0.1,
                   help='weight for correlation loss term')
    p.add_argument('--loss_plot',    default='loss_curve.png')
    p.add_argument('--pred_plot',    default='pred_vs_true.png')
    args = p.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Device:', device)

    full_input = args.input_dim * 2
    Xf, Xr, y, ids = preprocess(
        args.csv, args.wt_dir, args.mut_dir, args.scores_dir, args.cache)

    # split by complex for train/test only
    unique_cids = np.unique(ids)
    train_cids, test_cids = train_test_split(unique_cids,
                                             test_size=args.test_size,
                                             random_state=42)

    mask_tr = np.isin(ids, train_cids)
    mask_te = np.isin(ids, test_cids)

    Xf_tr, Xr_tr, y_tr = Xf[mask_tr], Xr[mask_tr], y[mask_tr]
    Xf_te, Xr_te, y_te = Xf[mask_te], Xr[mask_te], y[mask_te]

    loader_tr = DataLoader(DdgDataset(Xf_tr, Xr_tr, y_tr),
                           batch_size=args.batch_size, shuffle=True)
    loader_te = DataLoader(DdgDataset(Xf_te, Xr_te, y_te),
                           batch_size=args.batch_size)

    model = DDGPredictor(full_input,
                         args.hidden_dim,
                         args.dropout,
                         args.num_hidden).to(device)

    model, train_l, test_l = train_model(
        model, loader_tr, loader_te, device,
        args.lr, args.epochs, args.weight_decay,
        args.patience, args.min_delta, args.win,
        args.lr_sched_pat, args.lr_factor, args.min_lr,
        corr_lambda=args.corr_lambda
    )

    plot_loss(train_l, test_l, args.loss_plot)
    # final plot and metrics
    pear, spear = evaluate(model, loader_te, device, args.pred_plot)
    print(f'Final Test Pearson: {pear:.4f}, Spearman: {spear:.4f}')


if __name__ == '__main__':
    main()
