import torch
import numpy as np
from .gene_tokenizer import GeneVocab
from .dataset import (
    tokenized_dict_dataset_to_huggingface_dataset,
    save_huggingface_dataset,
    load_huggingface_dataset,
    load_numpy_dataset,
)
from cellstory.utils import get_obs
import logging
from torch.utils.data import  random_split
# get logger
logger = logging.getLogger(__name__)


def prepare_dataloader(args):
    if args.model_task == "pretrain":
        train_dataloader, val_dataloader, gene_tokens,atac_vocab_size = prepare_pretrain_dataloader(args)
        return train_dataloader, val_dataloader,gene_tokens,  atac_vocab_size
    elif args.model_task == "finetune":
        train_dataloader, val_dataloader,gene_tokens,  atac_vocab_size = prepare_finetune_dataloader(args)
        return train_dataloader, val_dataloader,gene_tokens,  atac_vocab_size
    elif args.model_task == "inference":
        dataloader,gene_tokens,  atac_vocab_size = prepare_inference_dataloader(args)
        return dataloader,gene_tokens,  atac_vocab_size


def prepare_pretrain_dataloader(args):
    logger.info("Load pretrain dataset")
    dataset,gene_tokens,  atac_vocab_size = load_dataset(args)
    # prepare dataloader
    is_train = True
    logger.info("Convert pretrain dataset to dataloader")
    total_size = len(dataset)
    val_size = int(total_size * 0.01)
    train_size = total_size - val_size

    logger.info(f"划分数据集: {train_size} 训练样本, {val_size} 验证样本")
    if val_size % args.batch_size == 1:
        val_batch_size = args.batch_size+1
    else:
        val_batch_size = args.batch_size

    # 设置随机种子以确保分割的可重复性
    generator = torch.Generator().manual_seed(42)
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size], generator=generator)
    
    train_dataloader = create_dataloader(
        train_dataset,
        is_train=is_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_mem=args.pin_mem,
        dist_eval=args.dist_eval,
    )
    logger.info("创建验证 DataLoader")
    val_dataloader = create_dataloader(
        val_dataset,
        is_train=False,
        batch_size=val_batch_size,
        num_workers=args.num_workers,
        pin_mem=args.pin_mem,
        dist_eval=args.dist_eval,
    )
   
    return train_dataloader, val_dataloader, gene_tokens, atac_vocab_size


def prepare_finetune_dataloader(args):
    # TODO add celltype support
    logger.info("Load finetune dataset")
    if args.RNA_prediction:
        dataset,gene_tokens, atac_vocab_size = load_dataset(args)
        dataset_RNA = dataset[1]
        dataset_ATAC = dataset[0]
        is_train = True
        logger.info("Convert finetune dataset to dataloader")
        total_size = len(dataset_ATAC)
        train_dataloader_ATAC = create_dataloader(
            dataset_ATAC,
            is_train=is_train,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_mem=args.pin_mem,
            dist_eval=args.dist_eval,
        )
        
        train_dataloader_RNA = create_dataloader(
            dataset_RNA,
            is_train=is_train,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_mem=args.pin_mem,
            dist_eval=args.dist_eval,
        )
        logger.info("创建验证 DataLoader")
        train_dataloader = [train_dataloader_ATAC,train_dataloader_RNA]
    
        val_dataloader = []
        return train_dataloader, val_dataloader,gene_tokens,atac_vocab_size
    else:
        dataset,gene_tokens,  atac_vocab_size = load_dataset(args)
        # prepare dataloader
        is_train = True
        logger.info("Convert finetune dataset to dataloader")
        total_size = len(dataset)
        val_size = int(total_size * 0.01)
        train_size = total_size - val_size

        logger.info(f"划分数据集: {train_size} 训练样本, {val_size} 验证样本")
        val_batch_size = args.batch_size
        while val_size % val_batch_size == 1:
            
            val_batch_size = val_batch_size+1
    
            

        # 设置随机种子以确保分割的可重复性
        generator = torch.Generator().manual_seed(42)
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size], generator=generator)
        
        train_dataloader = create_dataloader(
            train_dataset,
            is_train=is_train,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_mem=args.pin_mem,
            dist_eval=args.dist_eval,
        )
        logger.info("创建验证 DataLoader")
        val_dataloader = create_dataloader(
            val_dataset,
            is_train=False,
            batch_size=val_batch_size,
            num_workers=args.num_workers,
            pin_mem=args.pin_mem,
            dist_eval=args.dist_eval,
        )
        return train_dataloader, val_dataloader,gene_tokens,  atac_vocab_size


def prepare_inference_dataloader(args):
    logger.info("Load inference dataset")
    dataset,gene_tokens,  atac_vocab_size = load_dataset(args)
    # prepare dataloader
    is_train = False
    logger.info("Convert inference dataset to dataloader")
    dataloader = create_dataloader(
        dataset,
        is_train=is_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_mem=args.pin_mem,
        dist_eval=args.dist_eval,
    )
    return dataloader,gene_tokens, atac_vocab_size


def load_dataset(args):
    # load dataset with/o tokenization


    logger.info("Load ATAC dataset")
       
    dataset,gene_tokens, atac_vocab_size = load_atac_dataset(args)

        
    # finally, check vocab_size of RNA & ATAC modal

    return dataset,gene_tokens, atac_vocab_size







def load_atac_dataset(args):
    
    if args.atac_dataset_path is not None:
        atac_vocab = GeneVocab.from_file(args.atac_vocab_file)
        atac_vocab_size = len(atac_vocab)
     
        if args.cell_type_annotation or args.batch_correction:
            
            dataset,gene_tokens = load_numpy_dataset(args.atac_dataset_path,args.context_length+1,args.peak_length,args)
        if args.RNA_prediction:
            dataset_atac,gene_tokens = load_numpy_dataset(args.atac_dataset_path,args.context_length,args.peak_length,args)
            dataset_rna = load_huggingface_dataset(args.hvg_rna_dataset_path)
            dataset = [dataset_atac,dataset_rna]
        else:
            dataset,gene_tokens = load_numpy_dataset(args.atac_dataset_path,args.context_length,args.peak_length,args)
        # dataset = load_huggingface_dataset(args.atac_dataset_path)
        
        return dataset,gene_tokens, atac_vocab_size
    else:
        raise ValueError("Please tokenize anndata or provide atac_dataset_path")
    







def create_dataloader(
    dataset, is_train, batch_size, num_workers, pin_mem, dist_eval=False
):
    sampler = None
    
    return torch.utils.data.DataLoader(
        dataset,
        sampler=sampler,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_mem,
        drop_last=is_train,
        
        
    )
# collate_fn=merge_batch_tensors_by_dict_key,



def prepare_atac_inference_data(args):
    """
    prepare dataloader & obs for inference adata
    """
    assert args.atac_h5ad is not None
    logger.info("Prepare dataloader")
    dataloader,gene_tokens,  atac_vocab_size = prepare_dataloader(args)
    logger.info("Prepare obs")
    adata_obs = get_obs(args.atac_h5ad)

    return adata_obs, dataloader,gene_tokens,  atac_vocab_size

def prepare_atac_celltype_inference_data(args):
    """
    prepare dataloader & obs for inference adata
    """

    logger.info("Prepare dataloader")
    dataloader,gene_tokens,  atac_vocab_size = prepare_dataloader(args)
    
  

    return dataloader,gene_tokens,  atac_vocab_size

def merge_batch_tensors_by_dict_key(batch):
    """
    batch collate
    """
    batch_tensors = {}
    for tensor_key in batch[0]:
        if isinstance(batch[0][tensor_key], torch.Tensor):
            batch_tensors[tensor_key] = torch.stack([d[tensor_key] for d in batch])
        else:
            batch_tensors[tensor_key] = torch.tensor(
                [d[tensor_key] for d in batch], dtype=torch.long
            )
    return batch_tensors
