from datasets import Dataset, concatenate_datasets, load_dataset
from pathlib import Path
from typing import List, Dict, Union
import itertools
import logging
import re
import numpy as np
import pickle
from cellstory.preprocess import gene_tokenizer
# get logger
logger = logging.getLogger(__name__)


def save_data_list2d_to_parquet(data_list2d: List[List[Dict]], dataset_path: str):
    huggingface_ds = data_list2d_to_huggingface_dataset(data_list2d)
    save_huggingface_dataset_to_parquet(huggingface_ds, dataset_path)


def save_data_list2d_to_huggingface_dataset(
    data_list2d: List[List[Dict]], dataset_path: str
):
    huggingface_ds = data_list2d_to_huggingface_dataset(data_list2d)
    save_huggingface_dataset(huggingface_ds, dataset_path)


def data_list2d_to_huggingface_dataset(data_list2d: List[List[Dict]]):
    logger.info("Start converting dataset list2d to huggingface dataset")
    # first flat list
    flat_dataset = list(itertools.chain.from_iterable(data_list2d))
    # convert to huggingface dataset
    huggingface_ds = Dataset.from_list(flat_dataset)
    # set dataset format to pytorch for downstream use
    # logger.info("Set dataset format to pytorch for downstream use")
    # huggingface_ds.set_format(type="torch")
    logger.info("Finish converting dataset list2d to huggingface dataset")
    return huggingface_ds


# def tokenized_dict_dataset_to_huggingface_dataset(dict_dataset):
#     logger.info("Start converting dict dataset to huggingface dataset")
#     huggingface_ds = Dataset.from_list(list(dict_dataset.values()))
#     # set dataset format to pytorch for downstream use
#     logger.info("Set dataset format to pytorch for downstream use")
#     huggingface_ds.set_format(type="torch")
#     logger.info("Finish converting dict dataset to huggingface dataset")
#     return huggingface_ds
def tokenized_dict_dataset_to_huggingface_dataset(dict_dataset, batch_size=10000):
    logger.info("Start converting dict dataset to huggingface dataset")
    # 定义数据生成器函数，逐个生成 dict_dataset 中的值
    def data_generator(dict_dataset):
        for value in dict_dataset.values():
            yield value

    # 初始化空列表，用于存储分批创建的 Hugging Face 数据集对象
    huggingface_datasets = []
    # 初始化空列表，用于累积当前批次的数据
    current_batch = []
    # 遍历数据生成器中生成的数据
    for idx, data in enumerate(data_generator(dict_dataset)):
        current_batch.append(data)
        # 检查当前批次是否达到指定的 batch_size
        if len(current_batch) >= batch_size:
            # 创建一个 Hugging Face 数据集对象，从当前批次中的数据列表创建
            huggingface_batch = Dataset.from_list(current_batch)
            # 设置数据集对象的格式为 PyTorch 类型
            huggingface_batch.set_format(type="torch")
            # 将创建的数据集对象添加到 huggingface_datasets 列表中
            huggingface_datasets.append(huggingface_batch)
            # 记录处理的批次信息
            logger.info(f"Processed batch {len(huggingface_datasets)}")

            # 清空当前批次列表，释放内存
            current_batch = []

    # 处理最后一个不满 batch_size 的批次
    if current_batch:
        # 创建一个 Hugging Face 数据集对象，从当前批次中的数据列表创建
        huggingface_batch = Dataset.from_list(current_batch)
        # 设置数据集对象的格式为 PyTorch 类型
        huggingface_batch.set_format(type="torch")
        # 将创建的数据集对象添加到 huggingface_datasets 列表中
        huggingface_datasets.append(huggingface_batch)
        # 记录处理的批次信息
        logger.info(f"Processed batch {len(huggingface_datasets)}")

    # 记录合并所有批次成一个单一数据集的信息
    logger.info("Combining all batches into a single dataset")
    # 使用 concatenate_datasets 函数将 huggingface_datasets 列表中的所有数据集对象连接成一个单一数据集对象
    huggingface_ds = concatenate_datasets(huggingface_datasets)
    # 记录完成数据集转换的信息
    logger.info("Finish converting dict dataset to huggingface dataset")

    return huggingface_ds


def save_huggingface_dataset_to_parquet(dataset, parquet_path: Union[str, Path]):
    logger.info("Start saving huggingface dataset to parquet")
    dataset.to_parquet(parquet_path)
    logger.info("Finish saving huggingface dataset to parquet")


def save_huggingface_dataset(dataset, dataset_path: str):
    logger.info("Start saving huggingface dataset to disk")
    dataset.save_to_disk(dataset_path)
    logger.info("Finish saving huggingface dataset to disk")


def load_huggingface_dataset(dataset_path: Union[str, Path]):
    logger.info("Start loading huggingface dataset from disk")
    return Dataset.load_from_disk(dataset_path)
    logger.info("Finish loading huggingface dataset from disk")

def load_parquet_dataset(dataset_path: Union[str, Path]):
    logger.info("Start loading huggingface dataset from disk")
    cache_dir = Path(dataset_path) / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True) 
    parquet_files = [str(f) for f in Path(dataset_path).glob("*.parquet")]
    def sort_by_index(file_path):  
        # 使用正则表达式从文件名中提取起始和结束索引  
        match = re.search(r'(\d+)\.parquet$', file_path)  
        if match:  
            start = int(match.group(1))  
            # 为了确保按照起始索引排序，如果起始索引相同，则按结束索引排序  
            return start
        else:  
            # 如果文件名不符合预期格式，可以返回一个默认值或抛出异常  
            return float('inf') # 这里使用无穷大值确保不符合格式的文件排到最后  
    parquet_files = sorted(parquet_files, key=sort_by_index)  
    dataset = load_dataset(
        "parquet",
        data_files=parquet_files,
        cache_dir=str(cache_dir),
        split="train",
        
    )
    #  streaming = True, cache_dir=str(cache_dir),
    logger.info("Finish loading huggingface dataset from disk")
    return dataset



def load_numpy_dataset(dataset_path: Union[str, Path],context_length,peak_length,args):
    logger.info("Start loading huggingface dataset from disk")
    parquet_files = [str(f) for f in Path(dataset_path).glob("*.npy")]
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
    npy_files=npy_files[:-1] 
    if args.cell_type_annotation:
        with open(f"{dataset_path}/id2type.pkl", 'rb') as file:
            # 使用pickle.load()函数将读取的内容转换回Python对象
            cell_type_label = pickle.load(file)
        args.cell_type_number = len(cell_type_label.keys())
    if args.batch_correction:
        with open(f"{dataset_path}/id2type.pkl", 'rb') as file:
            # 使用pickle.load()函数将读取的内容转换回Python对象
            batch_label = pickle.load(file)
        args.batch_number = len(batch_label.keys())

        
    
    max_ends = {} 
    for npy in   npy_files:
        match = re.search(r'dataset_(\d+)_(\d+)_(\d+)\.npy$', npy) 
        data_num,start, end =int(match.group(1)), int(match.group(2)), int(match.group(3))
        if data_num not in max_ends:  
            max_ends[data_num] = end  
        else:  
            max_ends[data_num] = max(max_ends[data_num], end) 
    total_sum = sum(max_ends.values())  
    
    
    
    mm = np.memmap(f"{dataset_path}/large_data.bin", dtype=np.int8, mode='r+', shape = (total_sum, context_length,peak_length) )
    gene_tokens = np.load(f"{dataset_path}/gene_tokens.npy",allow_pickle=True)
    
    
    logger.info("Finish loading huggingface dataset from disk")
   
    return mm,gene_tokens

