#!/usr/bin/env python3
# File: MLP_attention.py

import os
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.model_selection import train_test_split
from scipy.stats import pearsonr, spearmanr
import matplotlib.pyplot as plt
from tqdm.auto import tqdm

class EarlyStopping:
    """
    Early stopping with moving average and minimum delta.
    """
    def __init__(self, patience=10, min_delta=1e-4, window_size=5):
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
            self.best_state = {k: v.cpu() for k, v in model.state_dict().items()}
            return False
        else:
            self.counter += 1
            return self.counter >= self.patience

class AttentionDataset(Dataset):
    """
    Loads per-mutation embeddings and per-residue scores,
    returns diff_fwd, diff_rev, scores (interface, burial), normalized ddG, and cid.
    """
    def __init__(self, csv_path, wt_dir, mut_dir, scores_dir):
        df = pd.read_csv(csv_path).dropna(subset=['ddG'])
        # load scores
        scores_dict = {}
        for fname in os.listdir(scores_dir):
            if not fname.endswith('.npz'): continue
            mut_id = fname[len("scores_"):-4]
            arr = np.load(os.path.join(scores_dir, fname))
            scores_dict[mut_id] = {'burial': arr['burial'], 'interface': arr['interface']}

        raw, y_vals = [], []
        for _, row in df.iterrows():
            mut_id, cid, y = row['#Pdb'], row['#Pdb_origin'], float(row['ddG'])
            wt_path = os.path.join(wt_dir,  f"{mut_id}_full_embeddings.npy")
            mt_path = os.path.join(mut_dir, f"{mut_id}_full_embeddings.npy")
            if not (os.path.exists(wt_path) and os.path.exists(mt_path)): continue
            sc = scores_dict.get(mut_id)
            if sc is None: continue
            emb = np.load(wt_path); L = emb.shape[0]
            if sc['burial'].shape[0] != L or sc['interface'].shape[0] != L: continue
            raw.append((wt_path, mt_path, sc['burial'], sc['interface'], y, cid))
            y_vals.append(y)
        if not raw:
            raise ValueError("No valid examples found. Check data or score files.")

        self.y_mean, self.y_std = np.mean(y_vals), np.std(y_vals)
        self.examples, self.ids = [], []
        for wt_path, mt_path, burial, interface, y, cid in raw:
            y_norm = (y - self.y_mean) / self.y_std
            self.examples.append((wt_path, mt_path, burial, interface, y_norm, cid))
            self.ids.append(cid)
        self.ids = np.array(self.ids)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        wt_path, mt_path, burial, interface, y_norm, cid = self.examples[idx]
        wt, mt = np.load(wt_path), np.load(mt_path)
        xf, xr = mt - wt, wt - mt
        sc = np.stack([interface, burial], axis=1)
        return {'xf': xf, 'xr': xr, 'scores': sc, 'y': y_norm, 'cid': cid}

def collate_fn(batch):
    B = len(batch)
    Lmax = max(d['xf'].shape[0] for d in batch)
    E = batch[0]['xf'].shape[1]
    xf = torch.zeros(B, Lmax, E)
    xr = torch.zeros(B, Lmax, E)
    sc = torch.zeros(B, Lmax, 2)
    mask = torch.zeros(B, Lmax, dtype=torch.bool)
    y = torch.zeros(B)
    for i, d in enumerate(batch):
        L = d['xf'].shape[0]
        xf[i,:L], xr[i,:L] = torch.from_numpy(d['xf']), torch.from_numpy(d['xr'])
        sc[i,:L] = torch.from_numpy(d['scores'])
        mask[i,:L] = 1
        y[i] = d['y']
    return xf, xr, sc, mask, y

class DDGPredictorAttention(nn.Module):
    """
    Attention-based pooling network predicting antisymmetric ddG,
    with score-gated residue embeddings.
    """
    def __init__(self, emb_dim=768, proj_dim=96, n_heads=2, n_layers=1, dropout=0.4):
        super().__init__()
        self.gate_layer  = nn.Linear(2,1)
        self.proj        = nn.Linear(emb_dim, proj_dim)
        self.score_proj  = nn.Linear(2, proj_dim, bias=False)
        enc_layer = nn.TransformerEncoderLayer(d_model=proj_dim,
                                               nhead=n_heads,
                                               dropout=dropout,
                                               batch_first=True)
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        self.pool_q      = nn.Parameter(torch.randn(proj_dim))
        self.head        = nn.Sequential(
            nn.LayerNorm(proj_dim),
            nn.Linear(proj_dim, proj_dim//2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(proj_dim//2, 1),
        )

    def _encode(self, xf, sc, mask):
        gate = torch.sigmoid(self.gate_layer(sc))  # B x L x 1
        xf   = xf * gate
        x    = self.proj(xf) + self.score_proj(sc)
        x    = self.transformer(x, src_key_padding_mask=~mask)
        att  = x.matmul(self.pool_q)               # B x L
        att  = att.masked_fill(~mask, float('-inf'))
        w    = F.softmax(att, dim=1).unsqueeze(-1)  # B x L x 1
        return (x * w).sum(dim=1)                   # B x proj_dim

    def forward(self, xf, xr, sc, mask):
        pf = self._encode(xf, sc, mask)
        pr = self._encode(xr, sc, mask)
        h  = (pf - pr) / 2
        return self.head(h).squeeze(-1)

def train_model(model, train_loader, val_loader, device,
                lr, epochs, weight_decay,
                patience, min_delta, window_size,
                lr_scheduler_patience, lr_factor, min_lr,
                corr_lambda=0.1):
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=lr_factor,
        patience=lr_scheduler_patience, verbose=True
    )
    criterion = nn.MSELoss()
    stopper = EarlyStopping(patience, min_delta, window_size)

    best_pear = -np.inf
    train_losses, val_losses = [], []

    for epoch in range(1, epochs+1):
        model.train()
        run_tr = 0.0
        for xf, xr, sc, mask, y in tqdm(train_loader, desc=f"Epoch {epoch}", leave=False):
            xf, xr, sc, mask, y = [t.to(device) for t in (xf, xr, sc, mask, y)]
            optimizer.zero_grad()

            pred = model(xf, xr, sc, mask)

            # weighted MSE
            w = torch.abs(y) + 1e-3
            mse_sample = (pred - y) ** 2
            loss_mse = (w * mse_sample).mean()

            # correlation loss (batch Pearson)
            p_center = pred - pred.mean()
            y_center = y    - y.mean()
            num = (p_center * y_center).mean()
            den = torch.sqrt((p_center**2).mean() * (y_center**2).mean()) + 1e-8
            pear_batch = num / den
            loss_corr = 1 - pear_batch

            loss = loss_mse + corr_lambda * loss_corr
            loss.backward()
            optimizer.step()

            # accumulate unweighted MSE for logging
            run_tr += mse_sample.mean().item() * xf.size(0)

        train_loss = run_tr / len(train_loader.dataset)
        train_losses.append(train_loss)

        # validation
        model.eval()
        run_val = 0.0
        ys, ps = [], []
        with torch.no_grad():
            for xf, xr, sc, mask, y in val_loader:
                xf, xr, sc, mask, y = [t.to(device) for t in (xf, xr, sc, mask, y)]
                pred = model(xf, xr, sc, mask)
                run_val += criterion(pred, y).item() * xf.size(0)
                ys.extend(y.cpu().numpy())
                ps.extend(pred.cpu().numpy())
        val_loss = run_val / len(val_loader.dataset)
        val_losses.append(val_loss)

        pear = pearsonr(ys, ps)[0]
        spear= spearmanr(ys, ps)[0]
        lr_cur = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch}: Train={train_loss:.4f}, Val={val_loss:.4f}, "
              f"Pearson={pear:.3f}, Spearman={spear:.3f} | LR={lr_cur:.2e}")

        # save best by Pearson
        if pear > best_pear:
            best_pear = pear
            torch.save(model.state_dict(), 'best_model.pt')

        scheduler.step(val_loss)
        if lr_cur < min_lr:
            print(f"LR below {min_lr:.1e}, stopping.")
            break
        if stopper(val_loss, model):
            print("Early stopping!")
            model.load_state_dict(stopper.best_state)
            break

    torch.save(model.state_dict(), 'final_model.pt')
    return model, train_losses, val_losses

def evaluate(model, loader, device, out_pred, y_mean, y_std):
    ys, ps = [], []
    model.eval()
    with torch.no_grad():
        for xf, xr, sc, mask, y in loader:
            xf, xr, sc, mask = [t.to(device) for t in (xf, xr, sc, mask)]
            pred = model(xf, xr, sc, mask).cpu().numpy()
            ys.extend(y.numpy())
            ps.extend(pred)
    pear  = pearsonr(ys, ps)[0]
    spear = spearmanr(ys, ps)[0]

    plt.figure(figsize=(6,6))
    plt.scatter(ys, ps, alpha=0.6)
    mn, mx = min(ys+ps), max(ys+ps)
    plt.plot([mn, mx], [mn, mx], '--')
    plt.xlabel('Normalized ddG')
    plt.ylabel('Normalized Pred')
    plt.title('True vs Pred (norm)')
    plt.text(0.05, 0.95, f'Pearson: {pear:.3f}\nSpearman: {spear:.3f}',
             transform=plt.gca().transAxes, va='top',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    plt.tight_layout()
    plt.savefig(out_pred)
    return pear, spear

def plot_loss(train_losses, val_losses, out):
    plt.figure(figsize=(8,5))
    epochs = range(1, len(train_losses)+1)
    plt.plot(epochs, train_losses, label='Train Loss')
    plt.plot(epochs, val_losses,   label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('MSE (norm)')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.tight_layout()
    plt.savefig(out)

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--csv',         default='data/SKEMPI2/SKEMPI2.csv')
    p.add_argument('--wt_dir',      default='data/SKEMPI2/SKEMPI2_cache/embedding_wildtype_full_2048')
    p.add_argument('--mut_dir',     default='data/SKEMPI2/SKEMPI2_cache/embedding_optimized_full_2048')
    p.add_argument('--scores_dir',  default='data/scores_cache')
    p.add_argument('--proj_dim',    type=int,   default=128)
    p.add_argument('--n_heads',     type=int,   default=4)
    p.add_argument('--n_layers',    type=int,   default=2)
    p.add_argument('--dropout',     type=float, default=0.3)
    p.add_argument('--lr',          type=float, default=1e-4)
    p.add_argument('--epochs',      type=int,   default=100)
    p.add_argument('--batch_size',  type=int,   default=8)
    p.add_argument('--test_size',   type=float, default=0.20)
    p.add_argument('--val_size',    type=float, default=0.10)
    p.add_argument('--weight_decay',type=float, default=1e-4)
    p.add_argument('--patience',    type=int,   default=10)
    p.add_argument('--min_delta',   type=float, default=1e-5)
    p.add_argument('--win',         type=int,   default=5)
    p.add_argument('--lr_sched_pat',type=int,   default=5)
    p.add_argument('--lr_factor',   type=float, default=0.5)
    p.add_argument('--min_lr',      type=float, default=1e-6)
    p.add_argument('--corr_lambda', type=float, default=0.05,
                     help='weight for correlation loss term')
    p.add_argument('--loss_plot',   default='loss_curve_att.png')
    p.add_argument('--pred_plot',   default='pred_vs_true_att.png')
    args = p.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Device:", device)

    # load dataset & split by complex id
    ds = AttentionDataset(args.csv, args.wt_dir, args.mut_dir, args.scores_dir)
    uids = np.unique(ds.ids)
    train_cids, test_cids = train_test_split(uids, test_size=args.test_size, random_state=42)
    train_cids, val_cids  = train_test_split(train_cids, test_size=args.val_size,   random_state=42)

    train_idx = np.where(np.isin(ds.ids, train_cids))[0]
    val_idx   = np.where(np.isin(ds.ids, val_cids))[0]
    test_idx  = np.where(np.isin(ds.ids, test_cids))[0]

    ds_tr  = Subset(ds, train_idx)
    ds_val = Subset(ds, val_idx)
    ds_te  = Subset(ds, test_idx)

    loader_tr = DataLoader(ds_tr, batch_size=args.batch_size, shuffle=True,
                           collate_fn=collate_fn, num_workers=1, pin_memory=True)
    loader_val= DataLoader(ds_val, batch_size=args.batch_size,
                           collate_fn=collate_fn, num_workers=1, pin_memory=True)
    loader_te = DataLoader(ds_te, batch_size=args.batch_size,
                           collate_fn=collate_fn, num_workers=1, pin_memory=True)

    model = DDGPredictorAttention(
        emb_dim=768,
        proj_dim=args.proj_dim,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        dropout=args.dropout
    ).to(device)

    model, train_losses, val_losses = train_model(
        model, loader_tr, loader_val, device,
        args.lr, args.epochs, args.weight_decay,
        args.patience, args.min_delta, args.win,
        args.lr_sched_pat, args.lr_factor, args.min_lr,
        corr_lambda=args.corr_lambda
    )

    plot_loss(train_losses, val_losses, args.loss_plot)

    pear, spear = evaluate(model, loader_te, device,
                           args.pred_plot,
                           ds.y_mean, ds.y_std)
    print(f"Test Pearson: {pear:.4f}, Spearman: {spear:.4f}")

if __name__ == '__main__':
    main()
