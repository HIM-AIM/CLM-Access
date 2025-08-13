# Copyright (c) 2022 Microsoft
# Licensed under The MIT License [see LICENSE for details]

import math

import torch
import torch.nn.functional as F
from torch import nn

from torch.nn import LayerNorm
from einops import rearrange
from flash_attn import flash_attn_func
from flash_attn.flash_attn_interface import flash_attn_varlen_qkvpacked_func
from flash_attn.bert_padding import unpad_input, pad_input, index_first_axis

from models.network import Wrapper, set_split_position


def get_activation_fn(activation):
    if activation == "relu":
        return F.relu
    elif activation == "gelu":
        return F.gelu
    elif activation == "swish":
        return F.silu
    else:
        raise NotImplementedError


class MultiheadAttention(nn.Module):
    def __init__(
        self,
        args,
        embed_dim,
        num_heads,
        dropout=0.0,
    ):
        super().__init__()
        self.args = args
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        
        if embed_dim % num_heads != 0:
            raise ValueError(
                f"The hidden size ({embed_dim}) is not a multiple of the number of attention "
                f"heads ({num_heads})"
            )
        self.head_dim = embed_dim // num_heads
        self.scaling = self.head_dim**-0.5
        self.dropout_p = dropout
        
        self.k_proj = Wrapper(args, nn.Linear(embed_dim, embed_dim, bias=True))
        self.v_proj = Wrapper(args, nn.Linear(embed_dim, embed_dim, bias=True))
        self.q_proj = Wrapper(args, nn.Linear(embed_dim, embed_dim, bias=True))        
        self.dropout_module = torch.nn.Dropout(dropout)
        
        self.out_proj = Wrapper(
            args, nn.Linear(embed_dim, embed_dim, bias=True)
        )
    
    def reset_parameters(self):
        nn.init.xavier_uniform_(self.k_proj.weight, gain=1 / math.sqrt(2))
        nn.init.xavier_uniform_(self.v_proj.weight, gain=1 / math.sqrt(2))
        nn.init.xavier_uniform_(self.q_proj.weight, gain=1 / math.sqrt(2))
        nn.init.xavier_uniform_(self.out_proj.weight)
        nn.init.constant_(self.out_proj.bias, 0.0)
    
    def attention_ops(self, q, k, v, key_padding_mask=None, attn_mask=None, rel_pos=None, is_causal=False):
        batch_size = q.shape[0]
        seqlen = q.shape[1]
        nheads = self.num_heads
        qkv = torch.stack([q, k, v], dim=2)
        x = rearrange(qkv, 'b s three h d -> b s (three h d)')
        key_padding_mask = ~key_padding_mask 
        x_unpad, indices, cu_seqlens, max_s = unpad_input(x, key_padding_mask)
        x_unpad = rearrange(x_unpad, 'nnz (three h d) -> nnz three h d', three=3, h=nheads)
        output_unpad = flash_attn_varlen_qkvpacked_func(
            x_unpad, cu_seqlens, max_s, self.dropout_p if self.training else 0.0,
            softmax_scale=self.scaling, causal=False
        )
        output_unpad = rearrange(output_unpad, 'nnz h d -> nnz (h d)')
        context_layer = pad_input(output_unpad, indices, batch_size, seqlen)
        return context_layer

    def forward(
        self,
        query,
        key,
        value,
        key_padding_mask=None,
        attn_mask=None,
    ):        
   
        bsz, src_len, embed_dim = query.size()
        assert embed_dim == self.embed_dim, f"query dim {embed_dim} != {self.embed_dim}"

        key_bsz, src_len, _ = key.size()
        assert key_bsz == bsz, f"{query.size(), key.size()}"
        assert value is not None
        assert bsz, src_len == value.shape[:2]

        # bsz, src_len, self.encoder_dim
        q = self.q_proj(query)
        k = self.k_proj(key)
        v = self.v_proj(value)
        # q *= self.scaling
        # bsz, src_len, self.encoder_dim --> bsz, src_len, num_heads, self.encoder_dim // num_heads
        q = rearrange(q, 'b l (h d) -> b l h d', h=self.num_heads)
        k = rearrange(k, 'b l (h d) -> b l h d', h=self.num_heads)
        v = rearrange(v, 'b l (h d) -> b l h d', h=self.num_heads)
        
        # add for inference
        if not self.training:
            q = q.to(torch.float16)
            k = k.to(torch.float16)
            v = v.to(torch.float16)

        context_layer = self.attention_ops(q, k, v, key_padding_mask, attn_mask)
        # output proj
        # added for inference
        if not self.training:
            context_layer = context_layer.to(torch.float32)
        attn_out = self.dropout_module(self.out_proj(context_layer))

        return attn_out


class FeedForwardNetwork(nn.Module):
    def __init__(
        self,
        embed_dim,
        ffn_dim,
        activation_fn,
        dropout,
        activation_dropout=None,
        layernorm_eps=None,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.activation_fn = get_activation_fn(activation=str(activation_fn))
        self.dropout_module = torch.nn.Dropout(dropout)
        self.fc1 = nn.Linear(self.embed_dim, ffn_dim)
        self.fc2 = nn.Linear(ffn_dim, self.embed_dim)
    
    def forward(self, x):
        x_intermediate = self.dropout_module(self.activation_fn(self.fc1(x)))
        ffn_out = self.dropout_module(self.fc2(x_intermediate))
        return ffn_out
    
    def reset_parameters(self):
        self.fc1.reset_parameters()
        self.fc2.reset_parameters()


class EncoderLayer(nn.Module):
    def __init__(self, args):
        super().__init__()  # 调用父类的构造函数
        self.args = args  # 保存传入的配置参数
        self.embed_dim = args.encoder_embed_dim  # 设置嵌入维度
        self.ffn_dim = args.encoder_ffn_embed_dim  # 设置FFN的维度
        self.pre_norm = args.pre_norm

        self.ffn = Wrapper(
            args,
            self.build_ffn(
                self.embed_dim,
                self.args,
            ),
        )  # 构建传统的前馈网络
        self.self_attn = self.build_self_attention(self.embed_dim, args)  # 构建自注意力模块
        self.layer_norm = Wrapper(args, LayerNorm(self.embed_dim)) 
            
    def build_ffn(self, embed_dim, args):
        return FeedForwardNetwork(
            embed_dim,
            self.ffn_dim,
            args.activation_fn,
            args.dropout,
            args.activation_dropout,
            args.layernorm_eps,
        )  # 构建FFN

    def build_self_attention(self, embed_dim, args):
        # 构建一个多头自注意力模块
        return MultiheadAttention(
            args,
            embed_dim,  # 嵌入维度
            args.encoder_attention_heads,  # 注意力头数
            dropout=args.attention_dropout,  # dropout概率
        )

    def forward(self, x, encoder_padding_mask, attn_mask=None, split_position=None):
        # 前向传播函数
        if split_position is not None:
        
            assert self.args.multi 
            self.apply(set_split_position(split_position))  # 设置分割位置

        if attn_mask is not None:
            # 如果提供了注意力掩码
            attn_mask = attn_mask.masked_fill(attn_mask.to(torch.bool), -1e8)  # 将掩码中的True位置设置为极小值

        if self.pre_norm:
            # 执行自注意力操作
            x_norm = self.layer_norm(x)
            x_attn = self.self_attn(
                query=x_norm,
                key=x_norm,
                value=x_norm,
                key_padding_mask=encoder_padding_mask,  # 编码器填充掩码
                attn_mask=attn_mask,  # 注意力掩码
            )
            x = x + x_attn
            x = x + self.ffn(self.layer_norm(x))
        else:
            x_attn = self.self_attn(
                query=x,
                key=x,
                value=x,
                key_padding_mask=encoder_padding_mask,  # 编码器填充掩码
                attn_mask=attn_mask,  # 注意力掩码
            )
            x = self.layer_norm(x + x_attn)
            x = self.layer_norm(x + self.ffn(x))
        
        l_aux = None  # 辅助损失为None

        return x, l_aux  # 返回输出和辅助损失


class Encoder(nn.Module):
    def __init__(
        self,
        args,
        **kwargs
    ):
        self.args = args  # 保存配置参数
        super().__init__(**kwargs)  # 初始化nn.Module
        
        self.layers = nn.ModuleList([EncoderLayer(args) for _ in range(args.encoder_layers)])  # 初始化存放编码器层的列表
        self.num_layers = len(self.layers)  # 记录编码器层数
        self.gradient_checkpointing = False

    def forward(
            self,
            token_embeddings,
            encoder_padding_mask,
            attn_mask=None,
            return_all_hiddens=False,
            split_position=None,
            **kwargs
    ):
        if split_position is not None:
         
            assert self.args.multi  
            self.apply(set_split_position(split_position))  

        encoder_states = []  # 初始化编码器状态列表

        x = token_embeddings
        if return_all_hiddens:
            # 如果要返回所有隐藏层状态
            encoder_states.append(x)  # 添加当前状态

        # incremental_state 用于推断时序列生成
        l_aux = []  # 初始化辅助输出列表
        for _, layer in enumerate(self.layers):
            # 遍历每一层
            x, l_aux_i = layer(x, encoder_padding_mask, attn_mask, split_position)
            if return_all_hiddens:
                # 如果要返回所有隐藏层状态
                assert encoder_states is not None
                encoder_states.append(x)  # 添加当前层的输出
            l_aux.append(l_aux_i)  # 添加辅助输出

        return {
            "encoder_out": x,  # 编码器输出
            "encoder_embedding": token_embeddings,  # 编码器嵌入
            "encoder_padding_mask": encoder_padding_mask,  # 编码器填充掩码
            "encoder_states": encoder_states,  # 编码器状态列表
            "l_aux": l_aux,  # 辅助输出列表
        }


