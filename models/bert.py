# Copyright (c) 2022 Microsoft
# Licensed under The MIT License [see LICENSE for details]

import math
import torch
import torch.nn as nn

from models.encoder_flash import Encoder


class BERT(nn.Module):
    def __init__(self, args, **kwargs):
        """_summary_
        input: atac, rna
        Args:
            args (_type_): _description_
        """
        super().__init__()  # 调用父类构造函数
        self.args = args  # 保存传入的参数
   
        self.atac_embed = nn.Embedding(args.atac_vocab_size, args.encoder_embed_dim)  # 创建文本嵌入层
        
        self.value_embed = ContinuousValueEncoder( args.encoder_embed_dim,args.peak_length)
        self.norm = nn.LayerNorm(args.encoder_embed_dim)

        self.encoder = Encoder(args)
        self.emb_dim = args.encoder_embed_dim

    def forward(
            self,
            atac_tokens=None,
            values_atac=None,
            atac_padding_position=None,
            attn_mask=None,  # 注意力掩码
    ):
  
        x_a = self.norm(self.atac_embed(atac_tokens))
        x_v = self.value_embed(values_atac)
        scale_num = 8
        x = x_a + x_v
        encoder_padding_mask = atac_padding_position
        split_position = -1 
               
        
        encoder_out = self.encoder(
            token_embeddings=x,  # 源token为空
            encoder_padding_mask=encoder_padding_mask,  # 编码器padding掩码
            attn_mask=attn_mask,  # 注意力掩码
            split_position=split_position,  
        )
        encoder_out["split_position"] = split_position  

        return encoder_out  # 返回编码器的输出


class ContinuousValueEncoder(nn.Module):
    """
    Encode real number values to a vector using neural nets projection.
    """

    def __init__(self, d_model: int, peak_length: int,dropout: float = 0.1, max_value: int = 512):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.linear1 = nn.Linear(peak_length, d_model)
        self.activation = nn.SiLU()
        self.linear2 = nn.Linear(d_model, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.max_value = max_value
    
    def forward(self, x):
        """
        Args:
            x: Tensor, shape [batch_size, seq_len]
        """
        # expand last dimension
        # x = x.unsqueeze(-1)
        # clip x to [-inf, max_value]
        x = torch.clamp(x, max=self.max_value)
        x = self.activation(self.linear1(x))
        x = self.linear2(x)
        x = self.norm(x)
        return self.dropout(x)

