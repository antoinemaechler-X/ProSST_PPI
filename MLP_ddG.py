import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import pearsonr, spearmanr
import matplotlib.pyplot as plt
from tqdm import tqdm
import argparse
import json
import itertools
from concurrent.futures import ProcessPoolExecutor
import os
from typing import Dict, List, Tuple, Any
from datetime import datetime
from explore_centrality_interface.interface_score import interface_scores

def parse_args():
    """Parse command line arguments for model parameters."""
    parser = argparse.ArgumentParser(description='Train ddG predictor with antisymmetrical predictions')
    
    # Model architecture parameters
    parser.add_argument('--input_dim', type=int, default=768, help='Input dimension (embedding size)')
    parser.add_argument('--hidden_dim', type=int, default=256, help='Hidden layer dimension')
    parser.add_argument('--dropout_rate', type=float, default=0.2, help='Dropout rate')
    parser.add_argument('--num_hidden_layers', type=int, default=2, help='Number of hidden layers')
    
    # Training parameters
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--num_epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--patience', type=int, default=10, help='Early stopping patience')
    parser.add_argument('--min_delta', type=float, default=1e-4, help='Minimum change in validation loss to be considered as improvement')
    parser.add_argument('--window_size', type=int, default=5, help='Window size for moving average of validation loss')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--test_size', type=float, default=0.2, help='Test set size')
    parser.add_argument('--val_size', type=float, default=0.2, help='Validation set size')
    
    # Data parameters
    parser.add_argument('--csv_path', type=str, default="data/SKEMPI2/SKEMPI2.csv", help='Path to CSV file')
    parser.add_argument('--wt_dir', type=str, default="data/SKEMPI2/SKEMPI2_cache/embedding_wildtype_full_2048", help='Wildtype embeddings directory')
    parser.add_argument('--mut_dir', type=str, default="data/SKEMPI2/SKEMPI2_cache/embedding_optimized_full_2048", help='Mutant embeddings directory')
    parser.add_argument('--preprocessed_dir', type=str, default="preprocessed_data", help='Directory for preprocessed data')
    parser.add_argument('--pdb_dir', type=str, default="data/SKEMPI2/SKEMPI2_cache/wildtype", help='Directory containing PDB files')
    
    # Grid search parameters
    parser.add_argument('--grid_search', action='store_true', help='Run grid search')
    parser.add_argument('--num_workers', type=int, default=8, help='Number of parallel workers for grid search')
    parser.add_argument('--preprocess_only', action='store_true', help='Only preprocess data, then exit')
    
    return parser.parse_args()

class DDGPredictor(nn.Module):
    def __init__(self, input_dim=768, hidden_dim=256, dropout_rate=0.2, num_hidden_layers=2):
        """
        Initialize the ddG predictor model.
        
        Args:
            input_dim (int): Input dimension (embedding size)
            hidden_dim (int): Hidden layer dimension
            dropout_rate (float): Dropout rate
            num_hidden_layers (int): Number of hidden layers
        """
        super().__init__()
        
        # Build layers dynamically based on num_hidden_layers
        layers = []
        current_dim = input_dim
        
        for i in range(num_hidden_layers):
            next_dim = hidden_dim // (2 ** i) if i > 0 else hidden_dim
            layers.extend([
                nn.Linear(current_dim, next_dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            ])
            current_dim = next_dim
        
        # Final layer
        layers.append(nn.Linear(current_dim, 1))
        
        self.mlp = nn.Sequential(*layers)
    
    def forward(self, x_wt_mt, x_mt_wt):
        """
        Forward pass with antisymmetrical prediction.
        
        Args:
            x_wt_mt: WT→MT embeddings
            x_mt_wt: MT→WT embeddings
            
        Returns:
            Antisymmetrical ddG prediction: (ddG(WT→MT) - ddG(MT→WT))/2
        """
        # Predict both directions
        ddg_wt_mt = self.mlp(x_wt_mt)
        ddg_mt_wt = self.mlp(x_mt_wt)
        
        # Calculate antisymmetrical prediction
        return (ddg_wt_mt - ddg_mt_wt) / 2

def load_embeddings(wt_path, mut_path):
    """Load wildtype and mutant embeddings."""
    wt_emb = np.load(wt_path)
    mut_emb = np.load(mut_path)
    return wt_emb, mut_emb

def get_interface_scores(pdb_path: str, sequence: str, chain_id: str) -> np.ndarray:
    """
    Calculate interface scores for each residue in the sequence.
    
    Args:
        pdb_path: Path to the PDB file
        sequence: Amino acid sequence
        chain_id: Chain identifier
        
    Returns:
        Array of interface scores, one per residue
    """
    scores = []
    for i, aa in enumerate(sequence, 1):
        # Create a fake mutation for each residue (e.g., "RA1A" for residue A at position 1)
        mutation = f"{aa}{chain_id}{i}A"
        try:
            score = interface_scores(pdb_path, mutation)
            scores.append(score)
        except Exception as e:
            print(f"Warning: Could not calculate interface score for {mutation}: {str(e)}")
            scores.append(0.0)  # Use 0.0 as fallback
    return np.array(scores)

def process_embeddings(wt_emb, mut_emb, interface_scores_array):
    """
    Calculate difference and maxpool across residues, weighted by interface scores.
    
    Args:
        wt_emb: Wildtype embeddings
        mut_emb: Mutant embeddings
        interface_scores_array: Array of interface scores for each residue
    """
    # Check for NaN in input
    if np.isnan(wt_emb).any():
        print("NaN found in wildtype embedding")
        return None
    if np.isnan(mut_emb).any():
        print("NaN found in mutant embedding")
        return None
        
    # Calculate raw difference
    diff = mut_emb - wt_emb
    if np.isnan(diff).any():
        print("NaN found after difference calculation")
        return None
    
    # Ensure interface scores match the number of residues
    if len(interface_scores_array) != diff.shape[0]:
        print(f"Warning: Interface scores length ({len(interface_scores_array)}) doesn't match number of residues ({diff.shape[0]})")
        # Pad or truncate interface scores to match
        if len(interface_scores_array) < diff.shape[0]:
            interface_scores_array = np.pad(interface_scores_array, (0, diff.shape[0] - len(interface_scores_array)))
        else:
            interface_scores_array = interface_scores_array[:diff.shape[0]]
    
    # Multiply differences by interface scores (broadcasting across embedding dimensions)
    weighted_diff = diff * interface_scores_array[:, np.newaxis]
    
    # Maxpool across residues (axis 0 is the residue dimension)
    pooled = np.max(weighted_diff, axis=0)
    if np.isnan(pooled).any():
        print("NaN found after maxpooling")
        return None
        
    return pooled

class EarlyStopping:
    """Improved early stopping with moving average and minimum improvement threshold."""
    def __init__(self, patience=10, min_delta=1e-4, window_size=5):
        self.patience = patience
        self.min_delta = min_delta
        self.window_size = window_size
        self.counter = 0
        self.best_loss = float('inf')
        self.best_model_state = None
        self.val_losses = []
    
    def __call__(self, val_loss, model):
        # Add current loss to window
        self.val_losses.append(val_loss)
        if len(self.val_losses) > self.window_size:
            self.val_losses.pop(0)
        
        # Calculate moving average
        avg_loss = sum(self.val_losses) / len(self.val_losses)
        
        # Check if this is the best loss so far
        if avg_loss < self.best_loss - self.min_delta:
            self.best_loss = avg_loss
            self.counter = 0
            self.best_model_state = model.state_dict().copy()
            return False
        
        self.counter += 1
        if self.counter >= self.patience:
            return True
        
        return False

def process_single_entry(args):
    """Process a single entry for parallel preprocessing."""
    row, wt_dir, mut_dir, pdb_dir = args
    pdb_id = row['#Pdb']
    complex_id = row['#Pdb_origin']
    ddg = row['ddG']
    mutation = row['Mutation(s)_cleaned']
    
    # Skip if ddG is NaN
    if pd.isna(ddg):
        return None
    
    # Construct paths
    wt_path = Path(wt_dir) / f"{pdb_id}_full_embeddings.npy"
    mut_path = Path(mut_dir) / f"{pdb_id}_full_embeddings.npy"
    pdb_path = Path(pdb_dir) / f"{pdb_id}.pdb"
    
    if not (wt_path.exists() and mut_path.exists() and pdb_path.exists()):
        return None
    
    try:
        # Load and process embeddings for both directions
        wt_emb, mut_emb = load_embeddings(wt_path, mut_path)
        
        # Get sequence from mutation info (first residue of first mutation)
        first_mut = mutation.split(',')[0].strip()
        chain_id = first_mut[1]  # Get chain ID from mutation (e.g., 'A' from 'RA88A')
        
        # Get interface scores for the actual mutation
        try:
            # Use interface_scores for the actual mutation
            scores = interface_scores(str(pdb_path), [first_mut])
            if not scores:
                print(f"Warning: No interface scores returned for {pdb_id} mutation {first_mut}")
                return None
            interface_score_value = scores[0]  # Get the score for this mutation
            
            # Create a mask of 1s and 0s based on whether each residue is the mutation site
            residue_pos = int(first_mut[2:-1])  # Get position (e.g., 88 from 'RA88A')
            interface_scores_array = np.zeros(len(wt_emb))
            interface_scores_array[residue_pos - 1] = interface_score_value  # -1 because positions are 1-indexed
            
        except Exception as e:
            print(f"Error calculating interface scores for {pdb_id} mutation {first_mut}: {str(e)}")
            return None
        
        # Process WT→MT direction
        diff_wt_mt = process_embeddings(wt_emb, mut_emb, interface_scores_array)
        if diff_wt_mt is None:
            print(f"Skipping {pdb_id} - NaN in WT→MT embeddings")
            return None
            
        # Process MT→WT direction
        diff_mt_wt = process_embeddings(mut_emb, wt_emb, interface_scores_array)
        if diff_mt_wt is None:
            print(f"Skipping {pdb_id} - NaN in MT→WT embeddings")
            return None
            
        return {
            'X_wt_mt': diff_wt_mt,
            'X_mt_wt': diff_mt_wt,
            'y': ddg,
            'complex_id': complex_id
        }
    except Exception as e:
        print(f"Error processing {pdb_id}: {str(e)}")
        return None

def preprocess_and_save_data(csv_path, wt_dir, mut_dir, preprocessed_dir, pdb_dir, num_workers=16):
    """Preprocess data once and save it permanently using parallel processing."""
    preprocessed_path = Path(preprocessed_dir)
    preprocessed_path.mkdir(exist_ok=True)
    
    # Check if preprocessed data already exists
    if (preprocessed_path / "data_info.json").exists():
        print("Preprocessed data already exists. Use --preprocess_only to force reprocessing.")
        return
    
    print("Preprocessing data (this will be done only once)...")
    
    # Read CSV file
    df = pd.read_csv(csv_path)
    
    # Print ddG distribution statistics
    print("\nddG Distribution Statistics:")
    print(df['ddG'].describe())
    print(f"Number of positive ddG: {len(df[df['ddG'] > 0])}")
    print(f"Number of negative ddG: {len(df[df['ddG'] < 0])}")
    print(f"Number of zero ddG: {len(df[df['ddG'] == 0])}")
    
    # Plot ddG distribution
    plt.figure(figsize=(10, 6))
    plt.hist(df['ddG'], bins=50)
    plt.xlabel('ddG')
    plt.ylabel('Count')
    plt.title('Distribution of ddG values')
    plt.savefig(preprocessed_path / 'ddg_distribution.png')
    plt.close()
    
    # Check for NaN in ddG values
    if df['ddG'].isna().any():
        print("Warning: NaN values found in ddG column")
        print("Rows with NaN ddG:")
        print(df[df['ddG'].isna()])
    
    # Prepare arguments for parallel processing
    process_args = [(row, wt_dir, mut_dir, pdb_dir) for _, row in df.iterrows()]
    
    # Process entries in parallel
    results = []
    skipped_count = 0
    nan_count = 0
    
    print(f"\nProcessing {len(process_args)} entries using {num_workers} workers...")
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        for result in tqdm(executor.map(process_single_entry, process_args), total=len(process_args)):
            if result is None:
                skipped_count += 1
                continue
            results.append(result)
    
    if not results:
        print("No valid entries found after processing!")
        return
    
    # Unpack results
    X_wt_mt = np.array([r['X_wt_mt'] for r in results])
    X_mt_wt = np.array([r['X_mt_wt'] for r in results])
    y = np.array([r['y'] for r in results])
    complex_ids = np.array([r['complex_id'] for r in results])
    
    print(f"\nDataset statistics:")
    print(f"Total entries in CSV: {len(df)}")
    print(f"Skipped entries: {skipped_count}")
    print(f"Final dataset size: {len(X_wt_mt)}")
    
    # Save preprocessed data
    print("Saving preprocessed data...")
    np.save(preprocessed_path / "X_wt_mt.npy", X_wt_mt)
    np.save(preprocessed_path / "X_mt_wt.npy", X_mt_wt)
    np.save(preprocessed_path / "y.npy", y)
    np.save(preprocessed_path / "complex_ids.npy", complex_ids)
    
    # Save dataset info
    info = {
        'total_entries': len(df),
        'skipped_entries': skipped_count,
        'final_size': len(X_wt_mt),
        'input_dim': X_wt_mt.shape[1],
        'timestamp': datetime.now().strftime("%Y%m%d_%H%M%S")
    }
    with open(preprocessed_path / "data_info.json", 'w') as f:
        json.dump(info, f, indent=2)
    
    print("Preprocessing complete!")

def load_preprocessed_data(preprocessed_dir):
    """Load preprocessed data."""
    preprocessed_path = Path(preprocessed_dir)
    
    if not (preprocessed_path / "data_info.json").exists():
        raise ValueError("Preprocessed data not found. Run with --preprocess_only first.")
    
    print("Loading preprocessed data...")
    return (
        np.load(preprocessed_path / "X_wt_mt.npy"),
        np.load(preprocessed_path / "X_mt_wt.npy"),
        np.load(preprocessed_path / "y.npy"),
        np.load(preprocessed_path / "complex_ids.npy")
    )

def train_with_params(params, args, train_data, val_data, test_data, device):
    """Train model with given parameters (moved outside run_grid_search for pickling)."""
    X_wt_mt_train, X_mt_wt_train, y_train = train_data
    X_wt_mt_val, X_mt_wt_val, y_val = val_data
    X_wt_mt_test, X_mt_wt_test, y_test = test_data
    
    # Initialize model with current parameters
    model = DDGPredictor(
        input_dim=args.input_dim,
        hidden_dim=params['hidden_dim'],
        dropout_rate=params['dropout_rate'],
        num_hidden_layers=params['num_hidden_layers']
    ).to(device)
    
    # Create args object with current parameters
    current_args = argparse.Namespace(**{**vars(args), **params})
    
    # Train model
    model, train_losses, val_losses = train_model(
        model,
        (X_wt_mt_train, X_mt_wt_train, y_train),
        (X_wt_mt_val, X_mt_wt_val, y_val),
        device,
        current_args
    )
    
    # Evaluate on test set
    model.eval()
    with torch.no_grad():
        predictions = model(
            torch.FloatTensor(X_wt_mt_test).to(device),
            torch.FloatTensor(X_mt_wt_test).to(device)
        ).cpu().numpy().squeeze()
    
    # Calculate metrics
    metrics = {
        'mse': mean_squared_error(y_test, predictions),
        'r2': r2_score(y_test, predictions),
        'pearson_corr': pearsonr(y_test, predictions)[0],
        'spearman_corr': spearmanr(y_test, predictions)[0]
    }
    
    return {
        'params': params,
        'metrics': metrics,
        'train_losses': train_losses,
        'val_losses': val_losses
    }

def train_model(model, train_data, val_data, device, args):
    """Train the model with improved early stopping."""
    X_wt_mt_train, X_mt_wt_train, y_train = train_data
    X_wt_mt_val, X_mt_wt_val, y_val = val_data
    
    # Verify no NaN values
    if (np.isnan(X_wt_mt_train).any() or np.isnan(X_mt_wt_train).any() or 
        np.isnan(y_train).any() or np.isnan(X_wt_mt_val).any() or 
        np.isnan(X_mt_wt_val).any() or np.isnan(y_val).any()):
        raise ValueError("NaN values found in training or validation data")
    
    # Convert to tensors
    X_wt_mt_train = torch.FloatTensor(X_wt_mt_train).to(device)
    X_mt_wt_train = torch.FloatTensor(X_mt_wt_train).to(device)
    y_train = torch.FloatTensor(y_train).to(device)
    X_wt_mt_val = torch.FloatTensor(X_wt_mt_val).to(device)
    X_mt_wt_val = torch.FloatTensor(X_mt_wt_val).to(device)
    y_val = torch.FloatTensor(y_val).to(device)
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    
    early_stopping = EarlyStopping(
        patience=args.patience,
        min_delta=args.min_delta,
        window_size=args.window_size
    )
    
    train_losses = []
    val_losses = []
    
    for epoch in range(args.num_epochs):
        # Training
        model.train()
        optimizer.zero_grad()
        outputs = model(X_wt_mt_train, X_mt_wt_train)
        loss = criterion(outputs.squeeze(), y_train)
        loss.backward()
        optimizer.step()
        train_losses.append(loss.item())
        
        # Validation
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_wt_mt_val, X_mt_wt_val)
            val_loss = criterion(val_outputs.squeeze(), y_val)
            val_losses.append(val_loss.item())
        
        # Early stopping check
        if early_stopping(val_loss.item(), model):
            print(f"Early stopping at epoch {epoch}")
            break
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch}: Train Loss = {loss.item():.4f}, Val Loss = {val_loss.item():.4f}")
    
    # Restore best model
    if early_stopping.best_model_state is not None:
        model.load_state_dict(early_stopping.best_model_state)
    
    return model, train_losses, val_losses

def get_run_dir(base_dir="MLP_runs"):
    """Create and return a directory for the current run."""
    # Create base directory if it doesn't exist
    base_path = Path(base_dir)
    base_path.mkdir(exist_ok=True)
    
    # Create timestamped directory for this run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = base_path / timestamp
    run_dir.mkdir(exist_ok=True)
    
    return run_dir

def evaluate_model(model, test_data, device, run_dir):
    """Evaluate model on test data."""
    X_wt_mt_test, X_mt_wt_test, y_test = test_data
    
    # Verify no NaN values
    if np.isnan(X_wt_mt_test).any() or np.isnan(X_mt_wt_test).any() or np.isnan(y_test).any():
        raise ValueError("NaN values found in test data")
    
    X_wt_mt_test = torch.FloatTensor(X_wt_mt_test).to(device)
    X_mt_wt_test = torch.FloatTensor(X_mt_wt_test).to(device)
    
    model.eval()
    with torch.no_grad():
        predictions = model(X_wt_mt_test, X_mt_wt_test).cpu().numpy().squeeze()
    
    # Print prediction statistics
    print("\nPrediction Statistics:")
    print(f"Min prediction: {predictions.min():.4f}")
    print(f"Max prediction: {predictions.max():.4f}")
    print(f"Mean prediction: {predictions.mean():.4f}")
    print(f"Number of positive predictions: {np.sum(predictions > 0)}")
    print(f"Number of negative predictions: {np.sum(predictions < 0)}")
    
    # Calculate metrics
    mse = mean_squared_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)
    pearson_corr, pearson_p = pearsonr(y_test, predictions)
    spearman_corr, spearman_p = spearmanr(y_test, predictions)
    
    print(f"\nTest Results:")
    print(f"Test set size: {len(y_test)}")
    print(f"MSE: {mse:.4f}")
    print(f"R²: {r2:.4f}")
    print(f"Pearson correlation: {pearson_corr:.4f} (p-value: {pearson_p:.4e})")
    print(f"Spearman correlation: {spearman_corr:.4f} (p-value: {spearman_p:.4e})")
    
    # Save metrics
    metrics = {
        'mse': float(mse),
        'r2': float(r2),
        'pearson_corr': float(pearson_corr),
        'pearson_p': float(pearson_p),
        'spearman_corr': float(spearman_corr),
        'spearman_p': float(spearman_p),
        'test_size': len(y_test),
        'prediction_stats': {
            'min': float(predictions.min()),
            'max': float(predictions.max()),
            'mean': float(predictions.mean()),
            'num_positive': int(np.sum(predictions > 0)),
            'num_negative': int(np.sum(predictions < 0))
        }
    }
    
    with open(run_dir / 'metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    
    # Plot predictions vs actual
    plt.figure(figsize=(10, 8))
    plt.scatter(y_test, predictions, alpha=0.5)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    plt.xlabel('Actual ddG')
    plt.ylabel('Predicted ddG')
    plt.title(f'Predicted vs Actual ddG\nPearson: {pearson_corr:.3f}, Spearman: {spearman_corr:.3f}')
    plt.savefig(run_dir / 'predictions_vs_actual.png')
    plt.close()
    
    # Add histograms on the sides
    plt.figure(figsize=(12, 10))
    gs = plt.GridSpec(4, 4)
    
    # Main scatter plot
    ax_main = plt.subplot(gs[1:4, 0:3])
    ax_main.scatter(y_test, predictions, alpha=0.5)
    ax_main.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    ax_main.set_xlabel('Actual ddG')
    ax_main.set_ylabel('Predicted ddG')
    ax_main.set_title(f'Predicted vs Actual ddG\nPearson: {pearson_corr:.3f}, Spearman: {spearman_corr:.3f}')
    
    # Histogram of actual values
    ax_hist_x = plt.subplot(gs[0, 0:3])
    ax_hist_x.hist(y_test, bins=50, orientation='vertical')
    ax_hist_x.set_title('Distribution of Actual ddG')
    
    # Histogram of predictions
    ax_hist_y = plt.subplot(gs[1:4, 3])
    ax_hist_y.hist(predictions, bins=50, orientation='horizontal')
    ax_hist_y.set_title('Distribution of Predicted ddG')
    
    plt.tight_layout()
    plt.savefig(run_dir / 'predictions_with_distributions.png')
    plt.close()

def plot_grid_search_results(results, run_dir):
    """Plot grid search results."""
    # Extract metrics and parameters
    metrics = ['mse', 'r2', 'pearson_corr', 'spearman_corr']
    params = ['hidden_dim', 'num_hidden_layers', 'dropout_rate', 'learning_rate']
    
    # Create subplots for each metric
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes = axes.ravel()
    
    for i, metric in enumerate(metrics):
        ax = axes[i]
        
        # Group results by each parameter
        for param in params:
            values = []
            scores = []
            for result in results:
                values.append(result['params'][param])
                scores.append(result['metrics'][metric])
            
            ax.scatter(values, scores, alpha=0.5, label=param)
        
        ax.set_xlabel('Parameter Value')
        ax.set_ylabel(metric)
        ax.set_title(f'{metric} vs Parameter Values')
        ax.legend()
    
    plt.tight_layout()
    plt.savefig(run_dir / 'grid_search_results.png')
    plt.close()

def run_grid_search(args, data, device):
    """Run grid search over hyperparameters."""
    # Create run directory
    run_dir = get_run_dir()
    
    # Save grid search parameters
    with open(run_dir / 'grid_search_params.json', 'w') as f:
        json.dump(vars(args), f, indent=2)
    
    # Define parameter grid
    param_grid = {
        'hidden_dim': [128, 256, 512],
        'num_hidden_layers': [1, 2, 3],
        'dropout_rate': [0.1, 0.2, 0.3],
        'learning_rate': [0.0001, 0.001, 0.01],
        'patience': [5, 10, 15],
        'window_size': [3, 5, 7]
    }
    
    # Save parameter grid
    with open(run_dir / 'parameter_grid.json', 'w') as f:
        json.dump(param_grid, f, indent=2)
    
    # Generate all parameter combinations
    param_combinations = [dict(zip(param_grid.keys(), v)) for v in itertools.product(*param_grid.values())]
    
    # Prepare data
    X_wt_mt, X_mt_wt, y, complex_ids = data
    
    # Split data once for all runs
    train_complexes, test_complexes = train_test_split(
        np.unique(complex_ids), test_size=args.test_size, random_state=42
    )
    
    train_mask = np.isin(complex_ids, train_complexes)
    test_mask = ~train_mask
    
    X_wt_mt_train, X_mt_wt_train, y_train = X_wt_mt[train_mask], X_mt_wt[train_mask], y[train_mask]
    X_wt_mt_test, X_mt_wt_test, y_test = X_wt_mt[test_mask], X_mt_wt[test_mask], y[test_mask]
    
    # Create validation set
    X_wt_mt_train, X_wt_mt_val, X_mt_wt_train, X_mt_wt_val, y_train, y_val = train_test_split(
        X_wt_mt_train, X_mt_wt_train, y_train, test_size=args.val_size, random_state=42
    )
    
    # Prepare data tuples for parallel processing
    train_data = (X_wt_mt_train, X_mt_wt_train, y_train)
    val_data = (X_wt_mt_val, X_mt_wt_val, y_val)
    test_data = (X_wt_mt_test, X_mt_wt_test, y_test)
    
    # Run grid search in parallel
    results = []
    with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
        futures = [
            executor.submit(
                train_with_params,
                params,
                args,
                train_data,
                val_data,
                test_data,
                device
            ) for params in param_combinations
        ]
        for future in tqdm(futures, total=len(param_combinations), desc="Grid Search"):
            results.append(future.result())
    
    # Save all results
    with open(run_dir / 'all_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    # Find best model based on validation MSE
    best_result = min(results, key=lambda x: x['metrics']['mse'])
    
    # Save best model parameters and metrics
    with open(run_dir / 'best_model.json', 'w') as f:
        json.dump({
            'parameters': best_result['params'],
            'metrics': best_result['metrics']
        }, f, indent=2)
    
    # Plot results
    plot_grid_search_results(results, run_dir)
    
    return best_result

def main():
    # Parse command line arguments
    args = parse_args()
    
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Preprocess data if needed
    if args.preprocess_only:
        preprocess_and_save_data(args.csv_path, args.wt_dir, args.mut_dir, 
                               args.preprocessed_dir, args.pdb_dir, args.num_workers)
        return
    
    # Load preprocessed data
    data = load_preprocessed_data(args.preprocessed_dir)
    
    if args.grid_search:
        print("Running grid search...")
        best_result = run_grid_search(args, data, device)
        print("\nBest model parameters:")
        print(json.dumps(best_result['params'], indent=2))
        print("\nBest model metrics:")
        print(json.dumps(best_result['metrics'], indent=2))
    else:
        # Create run directory
        run_dir = get_run_dir()
        
        # Save run parameters
        with open(run_dir / 'run_params.json', 'w') as f:
            json.dump(vars(args), f, indent=2)
        
        # Single training run with provided parameters
        X_wt_mt, X_mt_wt, y, complex_ids = data
        
        # Create train/test split
        train_complexes, test_complexes = train_test_split(
            np.unique(complex_ids), test_size=args.test_size, random_state=42
        )
        
        train_mask = np.isin(complex_ids, train_complexes)
        test_mask = ~train_mask
        
        X_wt_mt_train, X_mt_wt_train, y_train = X_wt_mt[train_mask], X_mt_wt[train_mask], y[train_mask]
        X_wt_mt_test, X_mt_wt_test, y_test = X_wt_mt[test_mask], X_mt_wt[test_mask], y[test_mask]
        
        # Create validation set
        X_wt_mt_train, X_wt_mt_val, X_mt_wt_train, X_mt_wt_val, y_train, y_val = train_test_split(
            X_wt_mt_train, X_mt_wt_train, y_train, test_size=args.val_size, random_state=42
        )
        
        # Initialize model
        model = DDGPredictor(
            input_dim=args.input_dim,
            hidden_dim=args.hidden_dim,
            dropout_rate=args.dropout_rate,
            num_hidden_layers=args.num_hidden_layers
        ).to(device)
        
        # Print model architecture
        print("\nModel Architecture:")
        print(model)
        
        # Save model architecture
        with open(run_dir / 'model_architecture.txt', 'w') as f:
            f.write(str(model))
        
        # Train model
        print("\nTraining model...")
        model, train_losses, val_losses = train_model(
            model, 
            (X_wt_mt_train, X_mt_wt_train, y_train), 
            (X_wt_mt_val, X_mt_wt_val, y_val), 
            device,
            args
        )
        
        # Plot training history
        plt.figure(figsize=(10, 6))
        plt.plot(train_losses, label='Training Loss')
        plt.plot(val_losses, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training History')
        plt.legend()
        plt.savefig(run_dir / 'training_history.png')
        plt.close()
        
        # Evaluate on test set
        evaluate_model(model, (X_wt_mt_test, X_mt_wt_test, y_test), device, run_dir)
        
        # Save model
        torch.save({
            'model_state_dict': model.state_dict(),
            'args': args
        }, run_dir / 'model.pt')

if __name__ == "__main__":
    main() 