#!/usr/bin/env python3
import numpy as np
import re

def normalize_id(pid):
    """Normalize protein ID by removing special characters and converting to lowercase."""
    return re.sub(r'[._]', '_', pid.strip().lower())

def get_prediction_id(protein, mutation, file_path):
    """Convert protein and mutation to the format used in prediction files.
    
    Args:
        protein: PDB code (e.g., '6_1ACB' or '1CSE')
        mutation: Mutation string (e.g., 'LI38G')
        file_path: Path to predictions file (not used anymore, kept for compatibility)
        
    Returns:
        str: Prediction ID in format '{prefix}-{pdb}_{chain}'
    """
    # Extract chain from mutation (e.g., 'LI38G' -> 'I')
    chain = mutation[1]  # Second character is the chain
    
    # Split protein into prefix and pdb if it contains underscore
    parts = protein.split('_')
    if len(parts) > 1:
        # If protein is like '6_1ACB', use that prefix
        prefix = parts[0]
        pdb = parts[1]
    else:
        # If no prefix in protein name, use 0 as default
        prefix = '0'
        pdb = protein
    
    return f"{prefix}-{pdb}_{chain}"

def read_predictions(file_path):
    """Read flexibility predictions from file."""
    predictions = {}
    current_protein = None
    current_values = []
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('>'):
                if current_protein is not None:
                    predictions[normalize_id(current_protein)] = np.array(current_values)
                current_protein = line[1:]
                current_values = []
            elif line:
                values = [float(x) for x in line.replace(', ', ',').split(',') if x]
                current_values.extend(values)
    if current_protein is not None:
        predictions[normalize_id(current_protein)] = np.array(current_values)
    return predictions

def get_mutation_position(mutation):
    """Extract the residue position from a mutation string.
    
    Args:
        mutation: Mutation string (e.g., 'LI38A')
        
    Returns:
        int: 0-based index of the mutation (e.g., 37 for LI38A)
    """
    # Extract the position number (e.g., '38' from 'LI38A')
    pos = int(''.join(c for c in mutation[2:-1] if c.isdigit()))
    # Convert to 0-based index
    return pos - 1

def flexibility_score(wt_file, mt_file, mutation, min_change_threshold=1e-5):
    """Calculate flexibility change at the mutation position.
    
    Args:
        wt_file: Path to wildtype predictions file
        mt_file: Path to mutant predictions file
        mutation: Mutation string (e.g., '6_1ACB_LI38A')
        min_change_threshold: Minimum change to consider (skip unchanged chains)
    
    Returns:
        float: Flexibility change at mutation position (MT-WT)
    """
    # Get protein ID from mutation (assuming format like '6_1ACB_LI38A')
    parts = mutation.split('_')
    if len(parts) == 3:  # Format: prefix_pdb_mutation
        protein = f"{parts[0]}_{parts[1]}"  # Keep the prefix with the protein
        mut = parts[2]
    else:  # Format: pdb_mutation
        protein = parts[0]
        mut = parts[1]
    
    # Get prediction IDs using the prefix from the mutation string
    wt_id = get_prediction_id(protein, mut, wt_file)
    mt_id = get_prediction_id(protein, mut, mt_file)
    
    # Read predictions
    wt_predictions = read_predictions(wt_file)
    mt_predictions = read_predictions(mt_file)
    
    wt_id = normalize_id(wt_id)
    mt_id = normalize_id(mt_id)
    
    if wt_id not in wt_predictions or mt_id not in mt_predictions:
        raise ValueError(f"Protein {protein} not found in predictions (wt_id={wt_id}, mt_id={mt_id})")
    
    wt_values = wt_predictions[wt_id]
    mt_values = mt_predictions[mt_id]
    
    if len(wt_values) != len(mt_values):
        raise ValueError(f"Length mismatch for {protein}: WT={len(wt_values)}, MT={len(mt_values)}")
    
    # Get mutation position (0-based index)
    mut_pos = get_mutation_position(mut)
    if mut_pos >= len(wt_values):
        raise ValueError(f"Mutation position {mut_pos+1} out of range (protein length: {len(wt_values)})")
    
    # Calculate change at mutation position
    wt_val = wt_values[mut_pos]
    mt_val = mt_values[mut_pos]
    change = mt_val - wt_val
    
    # Skip if change is too small
    if abs(change) < min_change_threshold:
        return float('nan')
    
    return change

def flexibility_scores(wt_file, mt_file, mutations, min_change_threshold=1e-5):
    """Calculate flexibility change scores for multiple mutations.
    
    Args:
        wt_file: Path to wildtype predictions file
        mt_file: Path to mutant predictions file
        mutations: List of mutation strings
        min_change_threshold: Minimum change to consider (skip unchanged chains)
    
    Returns:
        list: Maximum flexibility changes for each mutation
    """
    return [flexibility_score(wt_file, mt_file, mut, min_change_threshold) 
            for mut in mutations] 