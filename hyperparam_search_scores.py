#!/usr/bin/env python3
# File: hyperparam_search.py

import itertools
import json
import torch
import numpy as np
from sklearn.model_selection import train_test_split
from scipy.stats import pearsonr, spearmanr
from torch.utils.data import DataLoader

# import from your new script
from MLP_scores import train_model, DDGPredictor, preprocess, DdgDataset

def grid_search():
    # Hyperparameter grid
    hidden_dims     = [1024, 1536, 2048, 2560]
    num_hiddens     = [2, 3, 4, 5]
    dropouts        = [0.1, 0.2, 0.25, 0.3, 0.35]
    lrs             = [1e-5, 1e-4, 1e-3]
    weight_decays   = [1e-5, 1e-4]
    min_deltas      = [1e-5, 1e-4]

    param_combinations = list(itertools.product(
        hidden_dims, num_hiddens, dropouts, lrs, weight_decays, min_deltas
    ))

    # fixed args
    args = {
        'csv':         'data/SKEMPI2/SKEMPI2.csv',
        'wt_dir':      'data/SKEMPI2/SKEMPI2_cache/embedding_wildtype_full_2048',
        'mut_dir':     'data/SKEMPI2/SKEMPI2_cache/embedding_optimized_full_2048',
        'scores_dir':  'data/scores_cache',
        'cache':       'data/preproc_with_scores.npz',
        'input_dim':   768,    # per-channel dim; final is *2
        'epochs':      150,
        'batch_size':  32,
        'test_size':   0.20,
        'val_size':    0.10,
        'patience':    10,
        'win':         5,
        'lr_sched_pat':5,
        'lr_factor':   0.5,
        'min_lr':      1e-5,
    }

    # Pre-load / cache features once
    print("Preprocessing all data …")
    Xf, Xr, y, ids = preprocess(
        args['csv'],
        args['wt_dir'],
        args['mut_dir'],
        args['scores_dir'],
        args['cache']
    )
    # split train/test on mutation IDs, then train/val
    train_ids, test_ids = train_test_split(
        np.unique(ids),
        test_size=args['test_size'],
        random_state=42
    )
    mask = np.isin(ids, train_ids)
    Xf_tr, Xr_tr, y_tr = Xf[mask], Xr[mask], y[mask]
    Xf_te, Xr_te, y_te = Xf[~mask], Xr[~mask], y[~mask]
    Xf_tr, Xf_val, Xr_tr, Xr_val, y_tr, y_val = train_test_split(
        Xf_tr, Xr_tr, y_tr,
        test_size=args['val_size'],
        random_state=42
    )

    loader_tr  = DataLoader(DdgDataset(Xf_tr, Xr_tr, y_tr),
                            batch_size=args['batch_size'], shuffle=True)
    loader_val = DataLoader(DdgDataset(Xf_val, Xr_val, y_val),
                            batch_size=args['batch_size'])
    loader_te  = DataLoader(DdgDataset(Xf_te, Xr_te, y_te),
                            batch_size=args['batch_size'])

    best_params   = None
    best_pearson  = -np.inf
    best_spearman = -np.inf
    results       = []

    for (hidden_dim, num_hidden, dropout, lr, weight_decay, min_delta) in param_combinations:
        print("\n▶ Testing:",
              f"hidden={hidden_dim}", f"layers={num_hidden}",
              f"dropout={dropout}", f"lr={lr}",
              f"wd={weight_decay}", f"min_delta={min_delta}"
        )

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        full_input_dim = args['input_dim'] * 2
        model = DDGPredictor(
            full_input_dim,
            hidden_dim,
            dropout,
            num_hidden
        ).to(device)

        # train
        model, train_losses, val_losses = train_model(
            model,
            loader_tr,
            loader_val,
            device,
            lr,                    # lr
            args['epochs'],        # epochs
            weight_decay,          # weight_decay
            args['patience'],      # patience
            min_delta,             # min_delta
            args['win'],           # window_size
            args['lr_sched_pat'],  # lr_scheduler_patience
            args['lr_factor'],     # lr_factor
            args['min_lr']         # min_lr
        )

        # evaluate on test set
        ys, ps = [], []
        model.eval()
        with torch.no_grad():
            for xf, xr, y_batch in loader_te:
                xf, xr = xf.to(device), xr.to(device)
                pred = model(xf, xr).squeeze().cpu().numpy()
                ys.extend(y_batch.numpy())
                ps.extend(pred)

        pearson  = pearsonr(ys, ps)[0]
        spearman = spearmanr(ys, ps)[0]
        print(f"→ Test Pearson={pearson:.4f}, Spearman={spearman:.4f}")

        # record
        results.append({
            'hidden_dim': hidden_dim,
            'num_hidden': num_hidden,
            'dropout':    dropout,
            'lr':         lr,
            'weight_decay': weight_decay,
            'min_delta':  min_delta,
            'pearson':    pearson,
            'spearman':   spearman
        })

        if pearson > best_pearson:
            best_pearson  = pearson
            best_spearman = spearman
            best_params   = results[-1]

    # dump to file
    with open('grid_search_results.json', 'w') as f:
        json.dump({
            'best':   best_params,
            'scores': results
        }, f, indent=2)

    print("\nBest hyperparameters:", best_params)

if __name__ == '__main__':
    grid_search()
