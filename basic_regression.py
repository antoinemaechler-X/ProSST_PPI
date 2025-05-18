import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import os
from pathlib import Path

def load_embeddings(complex_id, chain, embedding_type='optimized'):
    """Load embeddings for a specific complex and chain."""
    base_path = Path(f'data/SKEMPI2/SKEMPI2_cache/embedding_{embedding_type}_2048')
    # Find the file that matches the complex_id and chain
    pattern = f'*_{complex_id}_{chain}_embeddings.npy'
    matching_files = list(base_path.glob(pattern))
    if not matching_files:
        raise FileNotFoundError(f"No embedding file found for complex {complex_id} chain {chain}")
    return np.load(matching_files[0])

def mean_pool_embeddings(embeddings):
    """Mean pool embeddings along the residue dimension."""
    return np.mean(embeddings, axis=0)

def prepare_data():
    """Prepare the dataset for training."""
    # Load the SKEMPI2 dataset
    df = pd.read_csv('data/SKEMPI2/SKEMPI2.csv')
    print(f"Initial dataset size: {len(df)}")
    
    # Filter out rows where ddG is NaN
    df = df.dropna(subset=['ddG'])
    print(f"Dataset size after removing NaN ddG: {len(df)}")
    
    # Count mutations per row
    df['mutation_count'] = df['Mutation(s)_cleaned'].str.count(',') + 1
    mutations_per_row = df['mutation_count']
    print("\nMutations per row statistics:")
    print(f"Min mutations per row: {mutations_per_row.min()}")
    print(f"Max mutations per row: {mutations_per_row.max()}")
    print(f"Mean mutations per row: {mutations_per_row.mean():.2f}")
    print(f"Median mutations per row: {mutations_per_row.median()}")
    
    # Find and show the row with the most mutations
    max_row = df.loc[df['mutation_count'].idxmax()]
    print(f"\nRow with most mutations:")
    print(f"Complex: {max_row['#Pdb_origin']}")
    print(f"Mutations: {max_row['Mutation(s)_cleaned']}")
    print(f"Number of mutations: {max_row['mutation_count']}")
    print(f"ddG: {max_row['ddG']}")
    
    # Print distribution of mutation counts
    mutation_counts = df['mutation_count'].value_counts().sort_index()
    print("\nDistribution of mutation counts:")
    for count, num_rows in mutation_counts.items():
        print(f"{count} mutation(s): {num_rows} rows")
    
    # Get unique complex IDs for train/test split
    unique_complexes = df['#Pdb_origin'].unique()
    print(f"\nNumber of unique complexes: {len(unique_complexes)}")
    
    # Check which complexes have embeddings
    complexes_with_embeddings = set()
    missing_embeddings = []
    for complex_id in unique_complexes:
        missing = []
        try:
            load_embeddings(complex_id, 'A', 'optimized')
        except FileNotFoundError:
            missing.append(f"{complex_id}_A_optimized")
        try:
            load_embeddings(complex_id, 'B', 'optimized')
        except FileNotFoundError:
            missing.append(f"{complex_id}_B_optimized")
        try:
            load_embeddings(complex_id, 'A', 'wildtype')
        except FileNotFoundError:
            missing.append(f"{complex_id}_A_wildtype")
        try:
            load_embeddings(complex_id, 'B', 'wildtype')
        except FileNotFoundError:
            missing.append(f"{complex_id}_B_wildtype")
            
        if not missing:
            complexes_with_embeddings.add(complex_id)
        else:
            missing_embeddings.extend(missing)
    
    print(f"Number of complexes with all embeddings: {len(complexes_with_embeddings)}")
    print(f"Number of complexes missing embeddings: {len(unique_complexes) - len(complexes_with_embeddings)}")
    if missing_embeddings:
        print("\nMissing embeddings:")
        for missing in missing_embeddings:
            print(f"- {missing}")
    
    # Filter df to only include complexes with all embeddings
    df = df[df['#Pdb_origin'].isin(complexes_with_embeddings)]
    print(f"\nDataset size after filtering for complexes with embeddings: {len(df)}")
    
    train_complexes, test_complexes = train_test_split(
        list(complexes_with_embeddings), 
        test_size=0.05, 
        random_state=42
    )
    print(f"Number of training complexes: {len(train_complexes)}")
    print(f"Number of test complexes: {len(test_complexes)}")
    
    X_train, X_test = [], []
    y_train, y_test = [], []
    
    # Process each row in the dataset
    for _, row in df.iterrows():
        complex_id = row['#Pdb_origin']
        
        # Load embeddings for both chains
        try:
            # Load optimized embeddings
            chain_a_opt = load_embeddings(complex_id, 'A', 'optimized')
            chain_b_opt = load_embeddings(complex_id, 'B', 'optimized')
            
            # Load wildtype embeddings
            chain_a_wt = load_embeddings(complex_id, 'A', 'wildtype')
            chain_b_wt = load_embeddings(complex_id, 'B', 'wildtype')
            
            # Mean pool all embeddings
            features = np.concatenate([
                mean_pool_embeddings(chain_a_opt),
                mean_pool_embeddings(chain_b_opt),
                mean_pool_embeddings(chain_a_wt),
                mean_pool_embeddings(chain_b_wt)
            ])
            
            # Add to appropriate dataset based on complex_id
            if complex_id in train_complexes:
                X_train.append(features)
                y_train.append(row['ddG'])
            else:
                X_test.append(features)
                y_test.append(row['ddG'])
                
        except FileNotFoundError as e:
            print(f"Warning: {e}")
            continue
    
    X_train = np.array(X_train)
    X_test = np.array(X_test)
    y_train = np.array(y_train)
    y_test = np.array(y_test)
    
    print(f"\nFinal dataset shapes:")
    print(f"X_train shape: {X_train.shape}")
    print(f"y_train shape: {y_train.shape}")
    print(f"X_test shape: {X_test.shape}")
    print(f"y_test shape: {y_test.shape}")
    print(f"Total number of mutations used: {len(y_train) + len(y_test)}")
    print(f"Training set size: {len(y_train)}")
    print(f"Test set size: {len(y_test)}")
    
    return X_train, X_test, y_train, y_test

def train_and_evaluate():
    """Train the model and evaluate its performance."""
    # Prepare data
    X_train, X_test, y_train, y_test = prepare_data()
    
    # Train model
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f"Mean Squared Error: {mse:.4f}")
    print(f"R² Score: {r2:.4f}")
    
    return model

if __name__ == "__main__":
    model = train_and_evaluate() 