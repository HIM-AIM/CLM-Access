import os
import sys
import dotmap

import torch
import lightning as pl

import scanpy as sc
from tqdm import tqdm

# add code to pythonpath
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from models.bert_value import BertForPretrain as BertForPretrain_Value
from cellstory.preprocess.input import (
    prepare_atac_celltype_inference_data,
)
from cellstory.inference.inference_celltype import generate_atac_celltype_metrics
from cellstory.utils import convert_to_path
from cellstory.logger import init_logger

# import experiment
from configs.config_eval import ex




def model_infer_celltype_atac(model, dataloader,gene_tokens,pad_id,embedding_type):
    preds = []
    labels = []
    
    for batch_data in tqdm(dataloader):
        batch_value = batch_data[:, :-1,:].clone()
        batch_label = batch_data[:, -1, 0].reshape(-1, 1).clone()
        gene_token = torch.from_numpy(gene_tokens).repeat(len(batch_data), 1,1).squeeze().to(model.device) 
        padding_mask = gene_token.eq(pad_id).to(model.device) 
        

        with torch.no_grad():
            # 对模型进行前向计算
            # -----error----- TODO: fix, using visual tokens
            outputs = model.bert(
                atac_tokens=gene_token,
                values_atac=batch_value.to(model.device).float(),
                atac_padding_position=padding_mask,
                attn_mask=None,
            )
            if embedding_type == "cls":
                atac_feats = outputs["encoder_out"][:, 0, :]
            elif embedding_type == "avgpool":
                atac_feats_without_cls = outputs["encoder_out"][
                        :, 1:, :
                    ]  # 去除每个序列的第一个元素
                    # 如果padding_mask也需要相应调整
                padding_mask_without_cls = padding_mask[
                    :, 1:
                ]  # 同样去除每个序列的第一个元素
                padding_mask_without_cls_ = ~padding_mask_without_cls
                padding_mask_without_cls_ = torch.unsqueeze(
                    padding_mask_without_cls_, dim=2
                )
                repr_wopadding = atac_feats_without_cls * padding_mask_without_cls_
                atac_feats = torch.sum(repr_wopadding, dim=1) / torch.unsqueeze(
                    torch.sum(~padding_mask_without_cls, dim=1), dim=1
                )
 
            atac_feats =  model.cell_type_annotation(atac_feats)
            pred = atac_feats.argmax(1).cpu().numpy()
            
            preds.extend(pred)
            labels.extend(batch_label.view(-1))
       
    return preds,labels



def atac_inference(args):
    # init logger
    logger = init_logger(args)

    # 定义数据集和模型
    dataloader,gene_tokens,  atac_vocab_size = prepare_atac_celltype_inference_data(args)
    

    # set vocab_size for RNA & ATAC
    # let rna_vocab_size=rna_vocab_size
 
    # let atac_vocab_size=atac_vocab_size
    args.atac_vocab_size = atac_vocab_size
    logger.info(f"vocab size:  ATAC: {args.atac_vocab_size}")

    # load model checkpoint
    logger.info("loading the model parameters")
    model = BertForPretrain_Value.load_from_checkpoint(
        args.model_load_path, map_location="cpu", config=args
    )
    model = model.cuda()
    model.eval()
 
    # inference from dataloader
    preds,labels = model_infer_celltype_atac(
        model, dataloader, gene_tokens=gene_tokens,pad_id=args.pad_id,embedding_type = args.embedding_type
    )
    

   
    # return adata
    return preds,labels


@ex.automain
def main(_config):
    # load config
    args_ = dotmap.DotMap(_config)
    # set repeatable seed
    pl.seed_everything(args_.seed)

    # init logger
    logger = init_logger(args_)

    # path settings
    args_.dirpath = convert_to_path(args_.dirpath)

    # create output directory if not exists
    if not os.path.exists(args_.dirpath):
        os.makedirs(args_.dirpath)
    # create log directory if not exists
    if not os.path.exists(args_.log_dir):
        os.makedirs(args_.log_dir, exist_ok=True)

    # Start taskpecific logic
    if args_.task == "atac_celltype":
        
        logger.info("Start inference for ATAC")
        preds,labels = atac_inference(args_)
        logger.info("Finish inference for ATAC")
        logger.info("Start calculating metrics for ATAC")
        generate_atac_celltype_metrics(args_, labels,preds)
        logger.info("Finish calculating metrics for ATAC")
