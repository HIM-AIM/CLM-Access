# Copyright (c) 2022 Microsoft
# Licensed under The MIT License [see LICENSE for details]

import copy

import torch
import torch.nn as nn


def Wrapper(args, module, dim=1):
    if args.multi:
        return Network(module, dim=dim)
    return module


def set_split_position(position):
    def apply_fn(module):
        if hasattr(module, "split_position"):
            module.split_position = position

    return apply_fn


class Network(nn.Module):
    def __init__(self, module, dim=1):
        super().__init__()
        self.dim = dim
        self.atac = module
        
       

    def forward(self, x, **kwargs):
        return self.atac(x, **kwargs)
   


