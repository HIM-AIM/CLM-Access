import argparse
import sys
import gc
from pathlib import Path
import numpy as np
import random
import os
import re
import glob  
import matplotlib.pyplot as plt 
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from cellstory.preprocess import pretrain_dataset,gene_tokenizer
from datasets import Dataset
import scanpy as sc
import scglue
from datasets import  load_dataset,concatenate_datasets
from scipy.sparse import csr_matrix  
from tqdm import tqdm
def _parse_args():
    # argparse
    
    parser = argparse.ArgumentParser(
        description="The pre-training dataset is processed"
    )
    parser.add_argument(
        "--input_datasetlist",
        type=str,
        default="/t9k/mnt/code/CLM-access/dataset/datalist.csv",
        help="pretrain dataset list file",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="output_dir",
        help="Directory to save data ",
    )
  
    parser.add_argument(
    "--ATAC_vocab_file",
    type=str,
    default="/t9k/mnt/code/CLM-access/dataset/ATAC_vocabulary_with_special_2000_unified.json",
    help="File containing the gene vocabulary, default to None. If None, will "
    "use the default gene vocabulary from scFormer, which use HGNC gene symbols.",
    )
   
    parser.add_argument(
    '--all_nonzero_value_set_1',
    type=int,
    default=True
    )
  
    parser.add_argument(
    '--context_length',
    type=int,
    default=2000
    )
    parser.add_argument(
    '--peak_length',
    type=int,
    default=600
    )

    parser.add_argument(
    '--context_select',
    default="random"  # random or truncation
    )
    parser.add_argument(
        "--append_cls",
        default=True
    )

    parser.add_argument(
        "--preprocessing",
        default=True
    )
    parser.add_argument(
        "--tokenizer",
        default=True
    )
    parser.add_argument(
        "--all_peaks",
        default=True
    )
    args = parser.parse_args()
    return args


if __name__=="__main__":
    args = _parse_args()
    # 读取文件路径列表
    # 提取两列数据

    if args.preprocessing:
         
        dataset_file_list, data_types = pretrain_dataset.read_csv_and_extract_columns(args.input_datasetlist)
        # 打印文件路径列表
        all_target_values = []
        end_sum = []
        for i in range(0,len(dataset_file_list)):
            adata = sc.read_h5ad(dataset_file_list[i],backed="r")
            # adata = sc.read_h5ad(dataset_file_list[i])
    
            gene_vocab = args.ATAC_vocab_file
            # 选择以"chr"开头的列       
            vocab = gene_tokenizer.GeneVocab.from_file(gene_vocab)

            def natural_sort_key(item):
    # 使用正则表达式提取染色体编号和起始位置
                if ':' in item[0] and '-' in item[0]:  
                    chromosome, positions = item[0].split(':')
                    chr_num = pretrain_dataset.chr_to_num(chromosome)
                    start, end = positions.split('-') 
            
                    chromosome_number = int(chr_num)  # 染色体编号转换为整数
                    start_position = int(start)     # 起始位置转换为整数
                    return (chromosome_number, start_position)
                else:  
            # 如果文件名不符合预期格式，可以返回一个默认值或抛出异常  
                    return float('inf'), float('inf')  # 这里使用无穷大值确保不符合格式的文件排到最后  
     
            adata_var_index = adata.var.index
            sorted_adata_var_index = [i for i, _ in sorted(enumerate(adata_var_index), key=lambda x: natural_sort_key([x[1]]))]
           
            chr_start_end = [(pretrain_dataset.chr_to_num(s.split(':')[0]), int(s.split(':')[1].split('-')[0]), int(s.split(':')[1].split('-')[1])) for s in adata.var.index[sorted_adata_var_index]]  
        
 
                
            sorted_data = sorted(vocab.get_stoi().items(), key=natural_sort_key)
        
            indexes = [] 
            chroms = []
            starts = []  
            ends = []  
            
            for key, value in sorted_data:  
                if ':' in key and '-' in key:  
                    chromosome, positions = key.split(':')
                    chr_num = pretrain_dataset.chr_to_num(chromosome)
                    start, end = positions.split('-')  
                    chroms.append(chr_num)
                    indexes.append(value)  
                    starts.append(int(start))  
                    ends.append(int(end)) 
                # 在这里处理每个词条  
        
            # 打印排序后的字典  
            patch_indices,region_counts = pretrain_dataset.map_points_to_regions_and_get_indices(chr_start_end ,chroms ,starts,ends,indexes)
           
            n_obs = adata.n_obs
            step = 500
            all_target_value=[]
            if i == 0:
                an = 0
            else:
                an = 0
            for start in tqdm(range(an, n_obs, step)):
                end = start + step
                if end > n_obs:
                    end = n_obs
            # extract to memory

                ad_mem = adata[start:end].to_memory()[:, sorted_adata_var_index]
               
                # ad_mem.X[ad_mem.X > 0] = 1
                if not isinstance(ad_mem.X, csr_matrix):
                    ad_mem.X = csr_matrix(ad_mem.X)


                target_values,gene_tokens,vocab = pretrain_dataset.load_anndata(ad_mem,data_types[i],args,patch_indices,vocab)
              
                if args.tokenizer:
            
                    patch_data,gene_ids= gene_tokenizer.tokenize_batch_edit(
                                                    data = target_values,
                                                    gene_ids=gene_tokens,
                                                    pad_token_id = vocab["<pad>"],
                                                    max_len=args.context_length,
                                                    target_length=args.peak_length,
                                                    pad_value = -2,
                                                    append_cls =  args.append_cls,
                                                    all_nonzero_value_set_1 = args.all_nonzero_value_set_1,
                                                    cls_id = vocab["<cls>"]
                                                )
                
                    np.save(f'{args.output_dir}/dataset_{i}_{start}_{end}.npy', patch_data)  
                del ad_mem
            
          
            end_sum.append(end)
          
                
            np.save(f'{args.output_dir}/gene_tokens.npy',gene_ids)
            vocab.save_json(args.output_dir + f"/vocab_{data_types[i]}.json") 
            adata.file.close()   
    
    
    parquet_files = [str(f) for f in Path(args.output_dir).glob("*.npy")]
    def sort_by_index(file_path):  
        # 使用正则表达式从文件名中提取起始和结束索引  
        match = re.search(r'dataset_(\d+)_(\d+)_(\d+)\.npy$', file_path)  
        if match:  
            data_num,start, end =int(match.group(1)), int(match.group(2)), int(match.group(3))  
            # 为了确保按照起始索引排序，如果起始索引相同，则按结束索引排序  
            return (data_num,start, end)  
        else:  
            # 如果文件名不符合预期格式，可以返回一个默认值或抛出异常  
            return float('inf'), float('inf')  # 这里使用无穷大值确保不符合格式的文件排到最后  
    npy_files = sorted(parquet_files, key=sort_by_index) 
     
    max_ends = {} 
    for npy in   npy_files[:-1]:
        match = re.search(r'dataset_(\d+)_(\d+)_(\d+)\.npy$', npy) 
        data_num,start, end =int(match.group(1)), int(match.group(2)), int(match.group(3))
       
        if data_num not in max_ends:  
            max_ends[data_num] = end  
        else:  
            max_ends[data_num] = max(max_ends[data_num], end) 
    total_sum = sum(max_ends.values())  
    # print(npy_files[:-1])
    arrays = []   
 

    if not os.path.exists(f"{args.output_dir}/large_data.bin"):  
    # 如果文件不存在，则创建文件  
    # 注意：这里以写入模式打开文件，如果文件已存在，其内容会被清空  
        with open(f"{args.output_dir}/large_data.bin", 'wb') as bin_file:  
            # 对于.bin文件，你可能想要写入一些二进制数据  
            # 例如，写入一个简单的字节数组  
            pass
    
    # 使用numpy.memmap创建内存映射数组  
    # mode='r+' 表示文件应该以读写模式打开（如果文件不存在，将会报错）  
    # 你也可以使用 'r' 模式来只读访问 
    print("start build",len(npy_files[:-1]))
    
    mm = np.memmap(f"{args.output_dir}/large_data.bin", dtype=np.int8, mode='r+', shape = (total_sum, args.context_length,args.peak_length) )

   # 遍历目录中的所有.npy文件  
    print("start all")
    for index,filename in enumerate(npy_files[:-1]):  
        print("start:",index)
        match = re.search(r'dataset_(\d+)_(\d+)_(\d+)\.npy$', filename)  
        data_num,start, end =int(match.group(1)), int(match.group(2)), int(match.group(3))  
     
        # 构建文件的完整路径  
        # 读取文件  
        array = np.load(filename, mmap_mode="r")  
        # 将读取的数组添加到列表中
        mm[start+sum(end_sum[0:data_num]):end+sum(end_sum[0:data_num])]=array

        
        print("finish:",index)
    print(mm[0])
    print(mm.shape)
    mm.flush()
  
    

             
   


    
    


    

  
  
 



