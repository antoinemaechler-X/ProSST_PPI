Here's a professional English README template for your needs:

---

# Protein-Protein Interaction (PPI) Prediction Project

This document provides instructions to run the PPI prediction pipeline. Please follow these steps carefully.

## 📥 Data Preparation
1. Download the **"Protein-protein complexes"** dataset (Category 6) from:  
   http://www.pdbbind.org.cn/download.php  
2. Save the downloaded data to:  
   `/PPI_data/PP`

## 🛠️ Dependencies
- Python 3.7+
- PyTorch 1.8+
- CUDA-enabled GPU (Recommended)
- Required packages:  
  ```bash
  pip install -r requirements.txt
  ```

## 🚀 Pipeline Execution

### 1. Feature Extraction
```bash
python PPI_preprocess.py
```
**Key Parameters**:  
- `vocab_size`: Choose from [20, 128, 512, 1024, 2048, 4096] (modify in script)  
- Output: Structure embeddings saved in corresponding folders  

**⚠️ Note**: Some embeddings may fail due to CUDA memory errors. 

### 2. Feature Generation
```bash
python PPI_embedding.py
```
- Combines sequence and structure features  
- Output: Per-chain protein features saved in `/PPI_embedding`

### 3. Model Training
```bash
python PPI_trainmodel.py
```
- Trains the final prediction model  
- Training results and checkpoints will be automatically saved  

## 🧠 Configuration Tips
1. For first-time runs, start with smallest `vocab_size=20`
2. Recommended workflow:  
   ```bash
   vocab_size=20 → 128 → 512 → 1024 (test progressively)
   ```


## 📂 Directory Structure
```
├── PPI_data/
│   └── PP/           # Raw data
│   └── 20/           # Struct embeddings
├── PPI_embedding/    # Processed features
├── PPI_preprocess.py
├── PPI_embedding.py
└── PPI_trainmodel.py
```