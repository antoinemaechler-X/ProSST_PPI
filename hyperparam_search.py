#!/usr/bin/env python3
# File: grid_search_optimization.py

import itertools
import json
import torch
from MLP_backup import train_model, DDGPredictor, preprocess, DdgDataset
from torch.utils.data import DataLoader
import numpy as np
from sklearn.model_selection import train_test_split
from scipy.stats import pearsonr, spearmanr


# Function to perform grid search optimization
def grid_search():
    # Hyperparameter grid
    hidden_dims = [128, 256, 512, 1024]
    num_hiddens = [2, 3, 4]
    dropouts = [0.1, 0.2, 0.3]
    lrs = [1e-5, 1e-4, 1e-3]
    weight_decays = [1e-5, 1e-4]
    min_deltas = [1e-5, 1e-4]

    # Generate all combinations of the hyperparameters
    param_combinations = list(itertools.product(hidden_dims, num_hiddens, dropouts, lrs, weight_decays, min_deltas))

    best_params = None
    best_pearson = -np.inf
    best_spearman = -np.inf
    results = []

    # Iterate over all hyperparameter combinations
    for combination in param_combinations:
        hidden_dim, num_hidden, dropout, lr, weight_decay, min_delta = combination
        print(f"Running grid search with params: hidden_dim={hidden_dim}, num_hidden={num_hidden}, "
              f"dropout={dropout}, lr={lr}, weight_decay={weight_decay}, min_delta={min_delta}")
        
        # Paths and file names
        args = {
            'csv': 'data/SKEMPI2/SKEMPI2.csv',
            'wt_dir': 'data/SKEMPI2/SKEMPI2_cache/embedding_wildtype_full_2048',
            'mut_dir': 'data/SKEMPI2/SKEMPI2_cache/embedding_optimized_full_2048',
            'cache': 'data/preproc.npz',
            'input_dim': 768,
            'hidden_dim': hidden_dim,
            'num_hidden': num_hidden,
            'dropout': dropout,
            'lr': lr,
            'epochs': 150,
            'batch_size': 32,
            'test_size': 0.20,
            'val_size': 0.10,
            'weight_decay': weight_decay,
            'patience': 10,
            'min_delta': min_delta,
            'win': 5,
            'lr_factor': 0.5,
            'min_lr': 1e-5,
            'loss_plot': 'loss_curve.png',
            'pred_plot': 'pred_vs_true.png'
        }

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Load and preprocess the data
        Xf, Xr, y, ids = preprocess(args['csv'], args['wt_dir'], args['mut_dir'], args['cache'])
        train_ids, test_ids = train_test_split(np.unique(ids), test_size=args['test_size'], random_state=42)
        mask = np.isin(ids, train_ids)
        Xf_tr, Xr_tr, y_tr = Xf[mask], Xr[mask], y[mask]
        Xf_te, Xr_te, y_te = Xf[~mask], Xr[~mask], y[~mask]
        Xf_tr, Xf_val, Xr_tr, Xr_val, y_tr, y_val = train_test_split(
            Xf_tr, Xr_tr, y_tr, test_size=args['val_size'], random_state=42)

        loader_tr = DataLoader(DdgDataset(Xf_tr, Xr_tr, y_tr), batch_size=args['batch_size'], shuffle=True)
        loader_val = DataLoader(DdgDataset(Xf_val, Xr_val, y_val), batch_size=args['batch_size'])
        loader_te = DataLoader(DdgDataset(Xf_te, Xr_te, y_te), batch_size=args['batch_size'])

        # Initialize the model
        model = DDGPredictor(args['input_dim'], hidden_dim, args['dropout'], num_hidden).to(device)

        # Train the model
        model, train_losses, val_losses = train_model(
            model, loader_tr, loader_val, device,
            lr=args['lr'], epochs=args['epochs'], weight_decay=args['weight_decay'],
            patience=args['patience'], min_delta=args['min_delta'], window_size=args['win'],
            lr_factor=args['lr_factor'], min_lr=args['min_lr'],
            lr_scheduler_patience=5  # Add the missing lr_scheduler_patience argument
        )

        # Evaluate the model
        ys, ps = [], []
        model.eval()
        with torch.no_grad():
            for xf, xr, y in loader_te:
                xf, xr = xf.to(device), xr.to(device)
                pred = model(xf, xr).squeeze().cpu().numpy()
                ys.extend(y.numpy())
                ps.extend(pred)
        
        pearson, spearman = pearsonr(ys, ps)[0], spearmanr(ys, ps)[0]
        print(f"Pearson: {pearson:.4f}, Spearman: {spearman:.4f}")

        # Save results
        results.append({
            'params': combination,
            'pearson': pearson,
            'spearman': spearman
        })

        # Update best params if current model performs better
        if pearson > best_pearson:
            best_pearson = pearson
            best_spearman = spearman
            best_params = combination

    # Save the best parameters and results to a JSON file
    with open('grid_search_results.json', 'w') as f:
        json.dump({
            'best_params': best_params,
            'best_pearson': best_pearson,
            'best_spearman': best_spearman,
            'results': results
        }, f, indent=4)

    print(f"Best Hyperparameters: {best_params}")
    print(f"Best Pearson: {best_pearson:.4f}, Best Spearman: {best_spearman:.4f}")


if __name__ == '__main__':
    grid_search()
