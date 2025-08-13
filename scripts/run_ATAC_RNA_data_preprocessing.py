
import json  # 添加 JSON 模块导入
import scanpy as sc
import scglue
from datasets import  load_dataset
from datasets import Dataset
import argparse
import os
from tqdm import tqdm
import numpy as np
import snapatac2 as snap
import anndata
def tokenize(adata, output_dir):
   
    all_data_dict = {}  # 初始化存储所有行数据的大字典
    for i in tqdm(range(adata.shape[0]), desc="Processing cells", total=adata.shape[0]):
        row_data = adata[i, :].X.toarray()[0]  # 获取当前行的数据

        all_data_dict[i] = {"value" : row_data}  # 将当前行数据添加到大字典中

 
    # 重组数据结构为列式格式
    dataset = Dataset.from_dict({
        "index": list(all_data_dict.keys()),
        "value": [np.array(v["value"]).astype(np.float32) for v in all_data_dict.values()]
    })

    print("Start saving huggingface dataset to disk")

    dataset.save_to_disk(output_dir, max_shard_size="20GB", num_proc=1)
    print("Finish saving huggingface dataset to disk")


def _parse_args():
    # argparse
    
    parser = argparse.ArgumentParser(
        description="The pre-training dataset is processed"
    )
    parser.add_argument(
        "--input_RNA_h5ad",
        type=str,
        default="raw/RNA_all_data.h5ad",
        help="pretrain dataset list file",
    )
    parser.add_argument(
        "--input_ATAC_h5ad",
        type=str,
        default="ATAC_all_data.h5ad",
        help="pretrain dataset list file",
    )
    parser.add_argument(
        "--output_train_dir",
        type=str,
        default="train/RNA",
        help="Directory to save data ",
    )
    parser.add_argument(
        "--output_test_dir",
        type=str,
        default="test/RNA",
        help="Directory to save data ",
    )
    parser.add_argument(
        "--output_raw_dir",
        type=str,
        default="RNA_predict_data/raw",
        help="Directory to save data ",
    )
    parser.add_argument(
        "--vocab_dir",
        type=str,
        default="RNA_predict_data",
        help="Directory to save data ",
    )

    parser.add_argument(
    '--hvg',
    type=int,
    default=2000
    )

    parser.add_argument(
        "--normalize_total",
        type=float,
        default=True
    ) 

   
    args = parser.parse_args()
    return args


if __name__=="__main__":
    args = _parse_args()
    # 读取文件路径列表

            
    adata_RNA = sc.read_h5ad(args.input_RNA_h5ad)
    adata_ATAC = sc.read_h5ad(args.input_ATAC_h5ad)

    print(adata_RNA)
 
    if args.normalize_total:
        sc.pp.normalize_total(adata_RNA,target_sum = 1e4)
        

    if args.hvg:
        adata_RNA_hvg = adata_RNA.copy()
        sc.pp.log1p(adata_RNA_hvg)
        sc.pp.highly_variable_genes(adata_RNA_hvg,n_top_genes=args.hvg)
        
        # adata = adata[:, adata.var.highly_variable]
        adata_RNA = adata_RNA_hvg[:, adata_RNA_hvg.var.highly_variable]
        # adata.write_h5ad("/t9k/mnt/code/cellstory-main-v5-all_peak_numpy_RNA/ATAC-inference/RNA_predict_article/rna_test.h5ad")
    train_fraction = 0.7  # 80% 用于训练  
    test_fraction = 0.3   # 20% 用于测试  
    seed = 42  # 可以是任意整数
    np.random.seed(seed)
    indices = np.random.permutation(adata_RNA.n_obs)
    
    # 按比例划分训练测试索引
    train_size = int(len(indices) * train_fraction)
    train_indices = indices[:train_size]
    test_indices = indices[train_size:]

    # 将索引转换为numpy数组  
    train_indices = np.array(train_indices)  
    test_indices = np.array(test_indices)  

    # 分割数据  
    adata_RNA_train = adata_RNA[train_indices, :]  
    # adata_train = adata
    adata_RNA_test = adata_RNA[test_indices, :]  

    adata_ATAC_train = adata_ATAC[train_indices, :]  
    # adata_train = adata
    adata_ATAC_test = adata_ATAC[test_indices, :]  


    adata_RNA_train.write_h5ad(os.path.join(args.output_raw_dir, "rna_train.h5ad"))
    adata_RNA_test.write_h5ad(os.path.join(args.output_raw_dir, "rna_test.h5ad"))
    adata_ATAC_train.write_h5ad(os.path.join(args.output_raw_dir, "atac_train.h5ad"))
    adata_ATAC_test.write_h5ad(os.path.join(args.output_raw_dir, "atac_test.h5ad"))





    genes = adata_RNA_train.var.index.tolist()

    gene_dict = {gene: idx for idx, gene in enumerate(genes)}  # 创建字典映射
    with open(f"{args.vocab_dir}/gene_dict.json", "w") as f:
        json.dump(gene_dict, f)

    tokenize(adata_RNA_train, args.output_train_dir)
    tokenize(adata_RNA_test, args.output_test_dir)
  
    

  


    


    

  
  
 



