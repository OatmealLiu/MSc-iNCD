import math
from functools import partial

import torch
import torch.nn as nn

from torch.nn.init import trunc_normal_
import torch.nn.functional as F


class VisionTransformerWithReLU(nn.Module):

    def __init__(self, base_vit):
        super().__init__()
        self.base_vit = base_vit

    def forward(self, x, return_features=False):
        features = self.base_vit(x)
        features = F.relu(features)
        return features


def vit_base_relu(args):
    from . import vision_transformer as vits

    base_vit = vits.__dict__['vit_base']()

    path_model_state_dict = args.dino_pretrain_path
    state_dict = torch.load(path_model_state_dict, map_location='cpu')
    base_vit.load_state_dict(state_dict)
    base_vit = base_vit.to(args.device)

    model = VisionTransformerWithReLU(base_vit=base_vit)

    # ----------------------
    # HOW MUCH OF BASE MODEL TO FINETUNE
    # ----------------------
    for m in model.base_vit.parameters():
        m.requires_grad = False

    # Only finetune layers from block 'args.grad_from_block' onwards
    for name, m in model.base_vit.named_parameters():
        if 'block' in name:
            block_num = int(name.split('.')[1])
            if block_num >= args.grad_from_block:
                m.requires_grad = True
    model = model.to(args.device)
    return model

