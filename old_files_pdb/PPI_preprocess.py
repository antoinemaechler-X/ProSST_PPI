import os
import torch
import gc
import sys
from glob import glob
from prosst.structure.quantizer import PdbQuantizer

# 配置路径和参数
vocab_size = 2048
pdb_dir = os.path.join("PPI_data", "PP")
output_dir = os.path.join("PPI_data", str(vocab_size))

# 创建输出目录
os.makedirs(output_dir, exist_ok=True)

# 获取所有PDB文件
pdb_files = glob(os.path.join(pdb_dir, "*.pdb"))

# errors to not compute

error_file_path = os.path.join("errors.txt")

# Load previously logged errors
if os.path.exists(error_file_path):
    with open(error_file_path, "r") as ef:
        skipped_errors = set(line.strip() for line in ef.readlines())
else:
    skipped_errors = set()

def reset_cuda():
    """完全释放 CUDA 资源，确保后续任务不受影响"""
    torch.cuda.empty_cache()
    gc.collect()
    with torch.no_grad():
        torch.cuda.ipc_collect()
    torch.cuda.synchronize()

def restart_program():
    """重新启动程序"""
    python = sys.executable
    os.execl(python, python, *sys.argv)

# 固定 GPU 设备
# os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Processing vocab size: {vocab_size} on GPU {os.environ.get('CUDA_VISIBLE_DEVICES', 'Not set (all visible)')}")

# 初始化处理器
processor = PdbQuantizer(structure_vocab_size=vocab_size)

# 处理每个PDB文件
for pdb_path in pdb_files[:300]:
    filename = os.path.basename(pdb_path)
    protein_name = os.path.splitext(filename)[0]

    # to not compute again errors
    if protein_name in skipped_errors:
        print(f"Skipping {filename}: previously failed.")
        continue

    output_path = os.path.join(output_dir, f"{protein_name}.fasta")
    
    # 如果目标文件已经存在，则跳过
    if os.path.exists(output_path):
        print(f"Skipping {filename}: Output file already exists.")
        continue

    print(f"Processing file: {filename}...")
    
    try:
        # 处理文件
        result = processor(pdb_path, return_residue_seq=False)
        
        # 提取结构信息
        if filename not in result[str(vocab_size)]:
            print(f"Skipping {filename}: Key '{filename}' not found in result.")
            with open(error_file_path, "a") as ef:
                ef.write(f"{protein_name}\n")
            continue
        
        struct_data = result[str(vocab_size)][filename].get("struct")
        if struct_data is None:
            print(f"Skipping {filename}: 'struct' key not found.")
            with open(error_file_path, "a") as ef:
                ef.write(f"{protein_name}\n")
            continue
        
        # 创建FASTA内容
        fasta_content = f">{protein_name}\n{','.join(map(str, struct_data))}"
        
        # 写入文件
        with open(output_path, "w") as f:
            f.write(fasta_content)
        
        print(f"Successfully processed {filename}, saved to {output_path}")

    except KeyError as e:
        print(f"Skipping {filename}: KeyError encountered - {e}")
        with open(error_file_path, "a") as ef:
            ef.write(f"{protein_name}\n")
    except RuntimeError as e:
        if "CUDA error: an illegal memory access was encountered" in str(e):
            print(f"Critical CUDA error in {filename}. Deleting file and restarting...")
            os.remove(pdb_path)  # 删除出错的PDB文件
            with open(error_file_path, "a") as ef:
                ef.write(f"{protein_name}\n")
            restart_program()  # 重新启动程序
        else:
            print(f"Error processing {filename}: {str(e)}")
            with open(error_file_path, "a") as ef:
                ef.write(f"{protein_name}\n")
                
    except Exception as e:
        print(f"Unexpected error processing {filename}: {str(e)}")
        with open(error_file_path, "a") as ef:
            ef.write(f"{protein_name}\n")

print("Processing completed!")
