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
    prepare_rna_inference_data,
    prepare_atac_inference_data,
)
from cellstory.inference.inference_rna import append_to_obsm, generate_rna_metrics
from cellstory.inference.inference_atac import generate_atac_metrics
from cellstory.utils import convert_to_path
from cellstory.logger import init_logger

# import experiment
from configs.config_eval import ex




def model_infer_atac(model, dataloader, embedding_type,gene_tokens,pad_id,cell_type_annotation):
    atac_reprs = []
    for batch_data in tqdm(dataloader):
        if cell_type_annotation:
            batch_data = batch_data[:, :-1,:].clone()
       
     
        gene_token = torch.from_numpy(gene_tokens).repeat(len(batch_data), 1,1).squeeze().to(model.device) 
        padding_mask = gene_token.eq(pad_id).to(model.device) 
        # padding_mask = batch_data["padding_mask"].cuda()
        # for inference using raw tokens without masking

        with torch.no_grad():
            # 对模型进行前向计算
            # -----error----- TODO: fix, using visual tokens
            outputs = model.bert(
                atac_tokens=gene_token,
                values_atac=batch_data.to(model.device).float(),
                atac_padding_position=padding_mask,
                attn_mask=None,
            )

            atac_feats = outputs["encoder_out"]

            if embedding_type == "cls":
                cls_repr = atac_feats[:, 0, :]
                atac_features = cls_repr / cls_repr.norm(dim=-1, keepdim=True)
                atac_reprs.extend(atac_features.cpu())
            elif embedding_type == "avgpool":
                # cal the mean without paddings, divide the real length without padding
                atac_feats_without_cls = atac_feats[
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
                avg_repr = torch.sum(repr_wopadding, dim=1) / torch.unsqueeze(
                    torch.sum(~padding_mask_without_cls, dim=1), dim=1
                )
                atac_features = avg_repr / avg_repr.norm(dim=-1, keepdim=True)
                atac_reprs.extend(atac_features.cpu())
        # in ATAC analysis using SnapATAC2, remember to cast dtype to float64
    stacked_embeddings = torch.stack(atac_reprs, dim=0).double().numpy()
    return stacked_embeddings



def atac_inference(args):
    # init logger
    logger = init_logger(args)

    # 定义数据集和模型
    adata_obs, dataloader,gene_tokens, rna_vocab_size, atac_vocab_size = (
        prepare_atac_inference_data(args)
    )


    # let atac_vocab_size=atac_vocab_size
    args.atac_vocab_size = atac_vocab_size
    logger.info(f"vocab size: ATAC: {args.atac_vocab_size}")

    # load model checkpoint
    logger.info("loading the model parameters")
    model = BertForPretrain_Value.load_from_checkpoint(
        args.model_load_path, map_location="cpu", config=args,strict = False
    )
    model = model.cuda()
    model.eval()
    # inference from dataloader
    atac_model_embed = model_infer_atac(
        model, dataloader, embedding_type=args.embedding_type,gene_tokens=gene_tokens,pad_id=args.pad_id,cell_type_annotation=args.cell_type_annotation
    )

    # read h5ad
    adata = sc.read_h5ad(args.atac_h5ad)
    # save raw count to layer
    adata.layers[args.raw_layer_key] = adata.X.copy()
    obsm_key = args.obsm_key if args.obsm_key is not None else "cellstory_atac"
    append_to_obsm(adata, obsm_key, atac_model_embed)
    # return adata
    return adata


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


    elif args_.task == "atacmlm":
        args_.atac_h5ad = convert_to_path(args_.atac_h5ad)
        logger.info("Start inference for ATAC")
        inferred_adata = atac_inference(args_)
        logger.info("Finish inference for ATAC")
        logger.info("Start calculating metrics for ATAC")
        fig, metric_df = generate_atac_metrics(args_, inferred_adata)
        logger.info("Finish calculating metrics for ATAC")
