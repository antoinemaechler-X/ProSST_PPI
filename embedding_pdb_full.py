from pathlib import Path
from Bio.PDB import PDBParser, PDBIO
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForMaskedLM
import tempfile
import os

device = "cuda" if torch.cuda.is_available() else "cpu"

def get_chain_center(chain):
    """Calculate the center of mass of a chain."""
    coords = []
    for residue in chain:
        if residue.id[0] == ' ':  # Only consider standard residues
            for atom in residue:
                coords.append(atom.get_coord())
    if not coords:
        return None
    return np.mean(coords, axis=0)

def reassign_empty_chain_atoms(structure):
    """Reassign atoms from empty chain to the nearest chain."""
    # Find chains with IDs
    valid_chains = [chain for chain in structure.get_chains() if chain.id != ' ']
    if not valid_chains:
        return structure
    
    # Calculate centers for valid chains
    chain_centers = {}
    for chain in valid_chains:
        center = get_chain_center(chain)
        if center is not None:
            chain_centers[chain.id] = center
    
    # Find empty chain
    empty_chain = None
    for chain in structure.get_chains():
        if chain.id == ' ':
            empty_chain = chain
            break
    
    if empty_chain is None:
        return structure
    
    # Reassign atoms from empty chain to nearest valid chain
    for residue in empty_chain:
        if residue.id[0] == ' ':  # Only consider standard residues
            residue_center = np.mean([atom.get_coord() for atom in residue], axis=0)
            # Find nearest chain
            min_dist = float('inf')
            nearest_chain_id = None
            for chain_id, center in chain_centers.items():
                dist = np.linalg.norm(residue_center - center)
                if dist < min_dist:
                    min_dist = dist
                    nearest_chain_id = chain_id
            
            if nearest_chain_id is not None:
                # Move residue to nearest chain
                target_chain = structure[0][nearest_chain_id]
                empty_chain.detach_child(residue.id)
                target_chain.add(residue)
    
    return structure

def parse_pdb_sequence(pdb_file: str) -> str:
    """Parse PDB file and return amino acid sequence for the entire structure."""
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("protein", pdb_file)
    
    # Reassign atoms from empty chain
    structure = reassign_empty_chain_atoms(structure)
    
    # Only consider standard amino acid residues
    standard_aa = {
        'ALA': 'A', 'CYS': 'C', 'ASP': 'D', 'GLU': 'E', 'PHE': 'F',
        'GLY': 'G', 'HIS': 'H', 'ILE': 'I', 'LYS': 'K', 'LEU': 'L',
        'MET': 'M', 'ASN': 'N', 'PRO': 'P', 'GLN': 'Q', 'ARG': 'R',
        'SER': 'S', 'THR': 'T', 'VAL': 'V', 'TRP': 'W', 'TYR': 'Y'
    }
    
    residues = []
    for model in structure:
        for chain in model:
            for residue in chain:
                # Only consider standard residues (not water, ions, etc.)
                if residue.id[0] == ' ':
                    res_name = residue.get_resname()
                    if res_name in standard_aa:
                        residues.append(standard_aa[res_name])
    
    return "".join(residues)

def read_structure_sequence(fasta_file: str) -> list:
    """Read structure sequence from FASTA file."""
    with open(fasta_file, "r") as f:
        lines = f.readlines()
        if len(lines) < 2:
            raise ValueError(f"Invalid FASTA file format: {fasta_file}")
        structure_seq = [int(i) for i in lines[1].strip().split(",")]
    return structure_seq

def tokenize_structure_sequence(structure_sequence: list):
    """Encode secondary structure sequence."""
    shift_seq = [1] + [i + 3 for i in structure_sequence] + [2]
    return torch.tensor([shift_seq], dtype=torch.long).to(device)

@torch.no_grad()
def get_per_residue_embeddings(model, tokenizer, sequence: str, structure_sequence: list, layer: int = -1):
    """Get per-residue embeddings for the protein chain."""
    # Convert 3-letter codes to 1-letter if needed
    standard_aa = {
        'ALA': 'A', 'CYS': 'C', 'ASP': 'D', 'GLU': 'E', 'PHE': 'F',
        'GLY': 'G', 'HIS': 'H', 'ILE': 'I', 'LYS': 'K', 'LEU': 'L',
        'MET': 'M', 'ASN': 'N', 'PRO': 'P', 'GLN': 'Q', 'ARG': 'R',
        'SER': 'S', 'THR': 'T', 'VAL': 'V', 'TRP': 'W', 'TYR': 'Y'
    }
    
    # Clean sequence to only include standard amino acids
    cleaned_sequence = ""
    for aa in sequence:
        if aa in standard_aa.values():
            cleaned_sequence += aa
        elif aa in standard_aa:
            cleaned_sequence += standard_aa[aa]
    
    # Verify sequence length matches structure sequence
    if len(cleaned_sequence) != len(structure_sequence):
        print(f"Warning: Length mismatch - Sequence: {len(cleaned_sequence)}, Structure: {len(structure_sequence)}")
        print("Truncating sequence to match structure length")
        cleaned_sequence = cleaned_sequence[:len(structure_sequence)]
    
    # First tokenize the sequence
    tokenized = tokenizer(
        [cleaned_sequence], 
        return_tensors="pt", 
        padding=True,
        truncation=True,
        add_special_tokens=True  # Add [CLS] and [SEP]
    ).to(device)
    
    # Tokenize structure sequence
    ss_tensor = tokenize_structure_sequence(structure_sequence)
    
    # Verify input shapes
    print(f"Input shapes:")
    print(f"  Sequence tokens: {tokenized.input_ids.shape}")
    print(f"  Structure tokens: {ss_tensor.shape}")
    
    # Get model outputs
    outputs = model(
        input_ids=tokenized.input_ids,
        attention_mask=tokenized.attention_mask,
        ss_input_ids=ss_tensor,
        output_hidden_states=True
    )
    
    if layer >= len(outputs.hidden_states):
        raise ValueError(f"Layer {layer} is out of range. Model has {len(outputs.hidden_states)} layers.")
    
    # Get embeddings for the specified layer
    # Remove special tokens (first and last) and get per-residue embeddings
    embeddings = outputs.hidden_states[layer][0][1:-1].cpu().numpy()
    
    # Verify output shape
    print(f"Output embedding shape: {embeddings.shape}")
    if embeddings.shape[0] != len(structure_sequence):
        print(f"Warning: Output embedding length ({embeddings.shape[0]}) doesn't match structure sequence length ({len(structure_sequence)})")
    
    return embeddings

def main():
    # Model path
    model_path = "AI4Protein/ProSST-2048"
    
    # Input and output directories
    token_dir = "data/SKEMPI2/SKEMPI2_cache/token_wildtype_full_2048"
    pdb_dir = "data/SKEMPI2/SKEMPI2_cache/wildtype"
    output_dir = "data/SKEMPI2/SKEMPI2_cache/embedding_wildtype_full_2048"
    
    # Allow overriding directories via environment variables
    token_dir = os.environ.get('TOKEN_DIR', token_dir)
    pdb_dir = os.environ.get('PDB_DIR', pdb_dir)
    output_dir = os.environ.get('OUTPUT_DIR', output_dir)
    
    print(f"Using directories:")
    print(f"  Token dir: {token_dir}")
    print(f"  PDB dir: {pdb_dir}")
    print(f"  Output dir: {output_dir}")
    
    # Create output directory if it doesn't exist
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Load model and tokenizer
    print("Loading model...")
    model = AutoModelForMaskedLM.from_pretrained(model_path, trust_remote_code=True).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    
    # Get all FASTA files
    fasta_files = list(Path(token_dir).glob("*.fasta"))
    total_files = len(fasta_files)
    print(f"Found {total_files} files to process")
    
    # Process each file
    for i, fasta_file in enumerate(fasta_files, 1):
        # Extract PDB ID from filename (format: 0_1CSE_full.fasta)
        parts = fasta_file.stem.split('_')
        prefix = parts[0]  # Get 0 from 0_1CSE_full
        pdb_id = parts[1]  # Get 1CSE from 0_1CSE_full
        
        # Check if output file already exists
        output_path = Path(output_dir) / f"{fasta_file.stem}_embeddings.npy"
        if output_path.exists():
            continue
        
        print(f"\nProcessing file {i}/{total_files}: {fasta_file.name}")
        
        # Get corresponding PDB file
        pdb_file = Path(pdb_dir) / f"{prefix}_{pdb_id}.pdb"
        if not pdb_file.exists():
            print(f"Skipping {fasta_file.name} - PDB file not found: {pdb_file}")
            continue
        
        try:
            # Get amino acid sequence from PDB
            print(f"Reading PDB file {pdb_file}...")
            sequence = parse_pdb_sequence(str(pdb_file))
            
            # Get structure sequence from FASTA
            structure_sequence = read_structure_sequence(str(fasta_file))
            
            # Get embeddings
            print("\nCalculating embeddings...")
            embeddings = get_per_residue_embeddings(model, tokenizer, sequence, structure_sequence)
            
            # Save embeddings
            np.save(output_path, embeddings)
            print(f"Saved embeddings to {output_path}")
            print(f"Embedding shape: {embeddings.shape}")
            
        except Exception as e:
            print(f"Error processing {fasta_file.name}: {str(e)}")
            continue

if __name__ == "__main__":
    main() 