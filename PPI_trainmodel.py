import os
import re
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from tqdm import tqdm  # 用于进度条显示
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

# 1. 数据集定义
class PPIDataset(Dataset):
    def __init__(self, index_file, embedding_dir):
        """
        参数：
          index_file: 存储 index 信息的文件路径
          embedding_dir: 存储所有 chain embedding 的文件夹路径
        """
        self.data = []  # 每个元素为 (chain_embeddings, binding_label)
        print(f"读取 index 文件: {index_file}")
        with open(index_file, 'r') as f:
            lines = f.readlines()
        total_lines = len(lines)
        valid_count = 0
        skipped_count = 0
        for line in lines:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            pdb_code = parts[0]
            # 假设 binding 信息在第4列，如 "Kd=22.5nM" 或 "pKd=7.65"
            binding_info = parts[3]
            
            if binding_info.startswith("Kd="):
                m = re.search(r'Kd=([\d\.]+)([a-zA-Z]+)', binding_info)
                if m is None:
                    print(f"跳过 {pdb_code}: 无法解析 binding 信息 {binding_info}")
                    skipped_count += 1
                    continue
                value = float(m.group(1))
                unit = m.group(2)
                # 单位转换：将数值转换为 M（摩尔）
                if unit.lower() == "nm":
                    factor = 1e-9
                elif unit.lower() == "um":
                    factor = 1e-6
                elif unit.lower() == "pm":
                    factor = 1e-12
                elif unit.lower() == "fm":
                    factor = 1e-15
                else:
                    print(f"未知单位 {unit} in {binding_info}, 使用原值")
                    factor = 1.0
                kd_in_M = value * factor
                # 转换为 pKd 值（pKd = -log10(Kd in M)）
                binding_value = -np.log10(kd_in_M)
            elif binding_info.startswith("pKd="):
                m = re.search(r'pKd=([\d\.]+)', binding_info)
                if m is None:
                    print(f"跳过 {pdb_code}: 无法解析 binding 信息 {binding_info}")
                    skipped_count += 1
                    continue
                binding_value = float(m.group(1))
            else:
                print(f"跳过 {pdb_code}: binding 信息格式不识别 {binding_info}")
                skipped_count += 1
                continue

            # 加载该 PDB code 对应的所有 chain 的 embedding
            chain_embeddings = []
            for fname in os.listdir(embedding_dir):
                # 文件格式示例： "2tgp.ent_I.npy"
                if fname.startswith(pdb_code) and fname.endswith('.npy'):
                    emb = np.load(os.path.join(embedding_dir, fname))
                    chain_embeddings.append(emb)
            if len(chain_embeddings) == 0:
                print(f"跳过 {pdb_code}: 未找到对应 embedding 文件")
                skipped_count += 1
                continue
            chain_embeddings = np.stack(chain_embeddings, axis=0)  # (num_chains, 768)
            self.data.append((chain_embeddings, binding_value))
            valid_count += 1
        
        print(f"数据读取完成: 总行数 {total_lines}，有效数据点 {valid_count}，跳过 {skipped_count}")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        chains, label = self.data[idx]
        chains = torch.tensor(chains, dtype=torch.float32)
        label = torch.tensor(label, dtype=torch.float32)
        return chains, label

def collate_fn(batch):
    """
    针对每个 batch 的数据，进行 pad 操作，使得每个数据点的 chain 数目统一，
    同时生成 mask（1 表示有效 token，0 表示 pad 部分）
    """
    max_chains = max(item[0].shape[0] for item in batch)
    batch_chains, masks, labels = [], [], []
    for chains, label in batch:
        n = chains.shape[0]
        pad_len = max_chains - n
        if pad_len > 0:
            pad = torch.zeros(pad_len, chains.shape[1])
            chains_padded = torch.cat([chains, pad], dim=0)
            mask = torch.cat([torch.ones(n), torch.zeros(pad_len)])
        else:
            chains_padded = chains
            mask = torch.ones(n)
        batch_chains.append(chains_padded)
        masks.append(mask)
        labels.append(label)
    batch_chains = torch.stack(batch_chains, dim=0)  # (batch, max_chains, 768)
    masks = torch.stack(masks, dim=0)  # (batch, max_chains)
    labels = torch.stack(labels, dim=0)  # (batch,)
    return batch_chains, masks, labels

# 2. Transformer 模型定义
class PPITransformer(nn.Module):
    def __init__(self, embed_dim=768, num_heads=8, num_layers=4, dropout=0.1):
        super(PPITransformer, self).__init__()
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, dropout=dropout)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        # 使用全连接层将聚合后的表示映射到一个标量（预测 binding 能量）
        self.regressor = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.ReLU(),
            nn.Linear(embed_dim // 2, 1)
        )
    
    def forward(self, x, mask):
        # x: (batch, seq_len, embed_dim)
        # 转置为 (seq_len, batch, embed_dim) 以适配 Transformer 模块
        x = x.transpose(0, 1)
        # 生成 key_padding_mask: 位置为0的部分为 True
        key_padding_mask = (mask == 0)
        x = self.transformer(x, src_key_padding_mask=key_padding_mask)
        x = x.transpose(0, 1)  # 恢复为 (batch, seq_len, embed_dim)
        # 利用 mask 做均值池化：只对有效 chain 做平均
        mask_expanded = mask.unsqueeze(-1)  # (batch, seq_len, 1)
        x = (x * mask_expanded).sum(dim=1) / mask_expanded.sum(dim=1)
        out = self.regressor(x)
        return out.squeeze(1)

# 定义评估函数：计算 Spearman 相关系数，并返回真实标签和预测标签
def evaluate(loader, model, device, dataset_name=""):
    true_labels = []
    pred_labels = []
    model.eval()
    with torch.no_grad():
        for batch_chains, masks, labels in loader:
            batch_chains = batch_chains.to(device)
            masks = masks.to(device)
            labels = labels.to(device)
            outputs = model(batch_chains, masks)
            true_labels.extend(labels.cpu().numpy())
            pred_labels.extend(outputs.cpu().numpy())
    true_labels = np.array(true_labels)
    pred_labels = np.array(pred_labels)
    corr, p_val = spearmanr(true_labels, pred_labels)
    print(f"{dataset_name} Spearman 相关系数: {corr:.4f} (p-value: {p_val:.4e})")
    return true_labels, pred_labels

def plot_true_vs_pred(true_labels, pred_labels, dataset_name, save_path):
    plt.figure(figsize=(6, 6))
    plt.scatter(true_labels, pred_labels, alpha=0.6, edgecolors='b', label='Data points')
    # 绘制 y = x 的参考线
    min_val = min(true_labels.min(), pred_labels.min())
    max_val = max(true_labels.max(), pred_labels.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='Ideal')
    plt.xlabel("True Binding Energy")
    plt.ylabel("Predicted Binding Energy")
    plt.title(f"{dataset_name}: True vs Predicted Binding Energy")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"{dataset_name} 的图像已保存至 {save_path}")

# 3. 训练流程
def main():
    # 超参数设定
    batch_size = 16
    epochs = 100
    learning_rate = 1e-4
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 数据路径，根据实际情况修改
    index_file = "/workspace/ProSST/PPI_data/PP/index/INDEX_general_PP.2020"
    embedding_dir = "/workspace/ProSST/PPI_embeddings"
    
    print("构建数据集...")
    dataset = PPIDataset(index_file, embedding_dir)
    
    # 划分训练集、测试集和验证集：训练集70%，测试集15%，验证集15%
    total_size = len(dataset)
    train_size = int(0.7 * total_size)
    test_size = int(0.15 * total_size)
    val_size = total_size - train_size - test_size
    train_dataset, test_dataset, val_dataset = random_split(dataset, [train_size, test_size, val_size])
    print(f"数据集划分：训练集 {train_size}，测试集 {test_size}，验证集 {val_size}")
    
    print("构建 DataLoader...")
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    
    print("构建模型...")
    model = PPITransformer(embed_dim=768, num_heads=8, num_layers=4, dropout=0.1).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    print("开始训练...")
    for epoch in range(epochs):
        model.train()
        total_train_loss = 0
        for batch_chains, masks, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}", unit="batch"):
            batch_chains = batch_chains.to(device)
            masks = masks.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_chains, masks)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            total_train_loss += loss.item() * batch_chains.size(0)
        avg_train_loss = total_train_loss / len(train_dataset)
        
        # 测试阶段（在测试集上评估）
        model.eval()
        total_test_loss = 0
        with torch.no_grad():
            for batch_chains, masks, labels in test_loader:
                batch_chains = batch_chains.to(device)
                masks = masks.to(device)
                labels = labels.to(device)
                outputs = model(batch_chains, masks)
                loss = criterion(outputs, labels)
                total_test_loss += loss.item() * batch_chains.size(0)
        avg_test_loss = total_test_loss / len(test_dataset)
        
        print(f"Epoch {epoch+1}: 训练集 Loss = {avg_train_loss:.4f}，测试集 Loss = {avg_test_loss:.4f}")
    
    print("训练完成!")
    
    # 保存模型
    model_path = "ppi_transformer_model.pth"
    torch.save(model.state_dict(), model_path)
    print(f"模型已保存至 {model_path}")
    
    # 评估训练集和验证集，计算 Spearman 相关系数
    print("在训练集上评估...")
    train_true, train_pred = evaluate(train_loader, model, device, dataset_name="训练集")
    print("在验证集上评估...")
    val_true, val_pred = evaluate(val_loader, model, device, dataset_name="验证集")
    
    # 绘制并保存训练集的 true vs predicted 图像
    plot_true_vs_pred(train_true, train_pred, "训练集", "train_true_vs_pred.png")
    # 绘制并保存验证集的 true vs predicted 图像
    plot_true_vs_pred(val_true, val_pred, "验证集", "val_true_vs_pred.png")

if __name__ == '__main__':
    main()
