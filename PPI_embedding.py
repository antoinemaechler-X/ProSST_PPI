from pathlib import Path
from Bio.PDB import PDBParser
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForMaskedLM

device = "cuda" if torch.cuda.is_available() else "cpu"

# PDB_DIR = "/workspace/ProSST/PPI_data/PP"
# FASTA_DIR = "/workspace/ProSST/PPI_data/20"
# OUTPUT_DIR = "./PPI_embeddings"
PDB_DIR = "zero_shot/example_data"
FASTA_DIR = "zero_shot/example_data"
OUTPUT_DIR = "embeddings_test"

Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)  # 创建存储embedding的文件夹

def parse_pdb_chain_sequence(pdb_file: str) -> dict:
    """解析PDB文件，返回各链的氨基酸序列"""
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("protein", pdb_file)
    sequences = {}
    for model in structure:
        for chain in model:
            chain_id = chain.id
            residues = [residue.get_resname() for residue in chain if residue.id[0] == ' ']
            sequences[chain_id] = "".join(residues)
    return sequences

def load_ss_input(fasta_file: str):
    """从fasta文件加载ss_input序列"""
    try:
        with open(fasta_file, "r") as f:
            lines = f.readlines()
            if len(lines) < 2:
                return None  # 确保有足够的行数据
            ss_sequence = list(map(int, lines[1].strip().split(',')))
            return ss_sequence
    except Exception as e:
        print(f"Error reading {fasta_file}: {e}")
        return None
    
def load_ss_input_v2(fasta_file: str):
    """Load amino acid sequence from a standard FASTA file."""
    try:
        with open(fasta_file, "r") as f:
            lines = f.readlines()
            sequence_lines = [line.strip() for line in lines if not line.startswith(">")]
            aa_sequence = "".join(sequence_lines)
            return aa_sequence
    except Exception as e:
        print(f"Error reading {fasta_file}: {e}")
        return None


def tokenize_structure_sequence(structure_sequence: list):
    """编码二级结构序列"""
    shift_seq = [1] + [i + 3 for i in structure_sequence] + [2]
    return torch.tensor([shift_seq], dtype=torch.long).to(device)

@torch.no_grad()
def get_embeddings(model, tokenizer, sequences: dict, ss_input: list, layer: int = -1, max_length: int = 2048):
    """获取蛋白质链的embedding"""
    embeddings = {}
    for chain_id, aa_sequence in sequences.items():
        if len(aa_sequence) > max_length:
            aa_sequence = aa_sequence[:max_length]
        
        tokenized = tokenizer(
            [aa_sequence], 
            return_tensors="pt", 
            max_length=max_length,
            padding=True,
            truncation=True
        ).to(device)
        
        if ss_input and len(ss_input) == len(tokenized.input_ids[0]) - 2:
            ss_tensor = tokenize_structure_sequence(ss_input)
        else:
            ss_tensor = tokenize_structure_sequence([0] * (len(tokenized.input_ids[0]) - 2))
        
        outputs = model(
            input_ids=tokenized.input_ids,
            attention_mask=tokenized.attention_mask,
            ss_input_ids=ss_tensor,
            output_hidden_states=True
        )
        
        if layer >= len(outputs.hidden_states):
            raise ValueError(f"Layer {layer} is out of range. Model has {len(outputs.hidden_states)} layers.")
        
        emb = outputs.hidden_states[layer][0].mean(dim=0).cpu().numpy()
        embeddings[chain_id] = emb
    return embeddings

if __name__ == "__main__":
    model_path = "AI4Protein/ProSST-2048"
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForMaskedLM.from_pretrained(model_path, trust_remote_code=True).to(device)
    
    pdb_files = list(Path(PDB_DIR).glob("*.pdb"))
    print(pdb_files)
    for pdb_file in pdb_files:
        pdb_id = pdb_file.stem  # 获取文件名（不含后缀）
        fasta_file = Path(FASTA_DIR) / f"{pdb_id}.fasta"
        
        if not fasta_file.exists():
            print(f"Skipping {pdb_id} (no corresponding fasta file)")
            continue
        
        sequences = parse_pdb_chain_sequence(str(pdb_file))
        ss_input = load_ss_input_v2(str(fasta_file))
        
        if ss_input is None:
            print(f"Skipping {pdb_id} due to invalid SS input")
            continue
        
        embeddings = get_embeddings(model, tokenizer, sequences, ss_input)
        
        for chain_id, emb in embeddings.items():
            save_path = Path(OUTPUT_DIR) / f"{pdb_id}_{chain_id}.npy"
            np.save(save_path, emb)
            print(f"Saved embedding: {save_path}")
