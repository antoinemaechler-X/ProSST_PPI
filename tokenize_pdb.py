from pathlib import Path
from Bio.PDB import PDBParser, PDBIO
from prosst.structure.quantizer import PdbQuantizer
import tempfile
import os
import glob
import numpy as np

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

def parse_pdb_chain_sequence(pdb_file: str) -> dict:
    """Parse PDB file and return amino acid sequences for each chain."""
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("protein", pdb_file)
    
    # Reassign atoms from empty chain
    structure = reassign_empty_chain_atoms(structure)
    
    sequences = {}
    for model in structure:
        for chain in model:
            chain_id = chain.id
            if chain_id == ' ':  # Skip empty chain
                continue
            residues = [residue.get_resname() for residue in chain if residue.id[0] == ' ']
            if residues:  # Only add non-empty chains
                sequences[chain_id] = "".join(residues)
    return sequences

def save_structure_sequence(pdb_id: str, chain_id: str, sequence: str, output_dir: str):
    """Save structure sequence to a FASTA file."""
    output_path = Path(output_dir) / f"{pdb_id}_{chain_id}.fasta"
    with open(output_path, "w") as f:
        f.write(f">{pdb_id}_{chain_id}\n")
        f.write(sequence)

def extract_chain_pdb(pdb_file: str, chain_id: str) -> str:
    """Extract a single chain from PDB file and save to temporary file."""
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("protein", pdb_file)
    
    # Reassign atoms from empty chain
    structure = reassign_empty_chain_atoms(structure)
    
    # Create a temporary file
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.pdb')
    temp_file.close()
    
    # Create a new structure with only the specified chain
    class ChainSelector:
        def __init__(self, chain_id):
            self.chain_id = chain_id
        def accept_model(self, model):
            return True
        def accept_chain(self, chain):
            return chain.id == self.chain_id
        def accept_residue(self, residue):
            return True
        def accept_atom(self, atom):
            return True
    
    # Save the selected chain to the temporary file
    io = PDBIO()
    io.set_structure(structure)
    io.save(temp_file.name, ChainSelector(chain_id))
    
    return temp_file.name

def process_pdb_file(pdb_file: str, output_dir: str, processor: PdbQuantizer):
    """Process a single PDB file and save its structure sequences."""
    pdb_id = Path(pdb_file).stem
    
    # Parse PDB file to get chain information
    sequences = parse_pdb_chain_sequence(pdb_file)
    
    # Check number of chains
    if len(sequences) != 2:
        raise ValueError(f"Error in {pdb_id}: Expected exactly 2 chains, found {len(sequences)} chains: {list(sequences.keys())}")
    
    # Get the two chains in order of appearance
    chain_ids = list(sequences.keys())
    
    # Process each chain and save as A and B
    for i, original_chain_id in enumerate(chain_ids):
        # Map to A and B based on order
        new_chain_id = 'A' if i == 0 else 'B'
        
        output_file = Path(output_dir) / f"{pdb_id}_{new_chain_id}.fasta"
        
        # Skip if already processed
        if output_file.exists():
            print(f"Skipping {pdb_id} chain {new_chain_id} - already processed")
            continue
        
        print(f"\nProcessing {pdb_id} chain {original_chain_id} (saving as {new_chain_id})")
        
        # Extract chain to temporary file
        temp_pdb = extract_chain_pdb(pdb_file, original_chain_id)
        
        try:
            # Get structure sequence using prosst for this chain
            result = processor(temp_pdb, return_residue_seq=False)
            # Use the temporary filename to access the result
            temp_filename = os.path.basename(temp_pdb)
            chain_seq = result['2048'][temp_filename]['struct']
            
            # Convert to comma-separated string
            structure_seq_str = ",".join(map(str, chain_seq))
            
            # Save to file with new chain ID (A or B)
            save_structure_sequence(pdb_id, new_chain_id, structure_seq_str, output_dir)
            print(f"Saved structure sequence for {pdb_id} chain {new_chain_id}")
        finally:
            # Clean up temporary file
            os.unlink(temp_pdb)

def main():
    # Input and output directories
    input_dir = "data/SKEMPI2/SKEMPI2_cache/optimized"
    output_dir = "data/SKEMPI2/SKEMPI2_cache/token_optimized_2048"
    
    # Create output directory if it doesn't exist
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Initialize PdbQuantizer
    processor = PdbQuantizer(structure_vocab_size=2048)
    
    # Get all PDB files
    pdb_files = glob.glob(os.path.join(input_dir, "*.pdb"))
    total_files = len(pdb_files)
    
    print(f"Found {total_files} PDB files to process")
    
    # Process each PDB file
    for i, pdb_file in enumerate(pdb_files, 1):
        print(f"\nProcessing file {i}/{total_files}: {pdb_file}")
        try:
            process_pdb_file(pdb_file, output_dir, processor)
        except ValueError as e:
            print(f"Error processing {pdb_file}: {str(e)}")
            continue

if __name__ == "__main__":
    main() 