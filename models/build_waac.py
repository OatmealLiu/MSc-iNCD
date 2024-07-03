import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from functools import partial
from torch.nn.init import trunc_normal_

class LinearHeadPCC(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.last_layer = nn.Linear(in_dim, out_dim, bias=False)
        # self.apply(self._init_weights)

    # def _init_weights(self, m):
    #     if isinstance(m, nn.Linear):
    #         trunc_normal_(m.weight, std=.02)
    #         if isinstance(m, nn.Linear) and m.bias is not None:
    #             nn.init.constant_(m.bias, 0)

    def forward(self, x, inference=True):
        # Feature normalization
        x = F.normalize(x, dim=-1, p=2)
        if not inference:
            # Feature centralization
            x_mean = x.mean(dim=0)
            x_mean = x_mean.reshape((1, x_mean.size(0)))
            x -= x_mean
        # else:
        #     prototype = self.last_layer.weight.data.clone()
        #     mean_prototype = prototype.mean(dim=1)
        #     mean_prototype = mean_prototype.reshape((1, x.size(1)))
        #     x -= mean_prototype
        # forward
        x = self.last_layer(x)
        return x

    @torch.no_grad()
    def normalize_prototypes(self, s=1.0):
        # Centered cosine normalization (Pearson Correlation Coefficient)
        # 1st: normalize weights vectors
        w = self.last_layer.weight.data.clone()
        w = s * F.normalize(w, dim=1, p=2)

        # 2nd: centralize weights vectors
        w_mean = w.mean(dim=1)
        w_mean = w_mean.reshape(w_mean.size(0), 1)
        w -= w_mean
        # 3rd: update the PCCed weights vectors
        self.last_layer.weight.copy_(w)


class LinearHead(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.last_layer = nn.Linear(in_dim, out_dim, bias=False)
        # self.apply(self._init_weights)

    # def _init_weights(self, m):
    #     if isinstance(m, nn.Linear):
    #         trunc_normal_(m.weight, std=.02)
    #         if isinstance(m, nn.Linear) and m.bias is not None:
    #             nn.init.constant_(m.bias, 0)

    def forward(self, x, inference=True):
        x = F.normalize(x, dim=-1, p=2)
        x = self.last_layer(x)
        return x

    @torch.no_grad()
    def normalize_prototypes(self, s=1.0):
        # Cosine normalization
        w = self.last_layer.weight.data.clone()
        w = s * F.normalize(w, dim=1, p=2)
        self.last_layer.weight.copy_(w)


def build_backbone_vit(args):
    from . import vision_transformer as vits
    path_model_state_dict = args.dino_pretrain_path

    model = vits.__dict__['vit_base']()
    state_dict = torch.load(path_model_state_dict, map_location='cpu')
    model.load_state_dict(state_dict)
    model = model.to(args.device)

    # ----------------------
    # HOW MUCH OF BASE MODEL TO FINETUNE
    # ----------------------
    for m in model.parameters():
        m.requires_grad = False

    # Only finetune layers from block 'args.grad_from_block' onwards
    for name, m in model.named_parameters():
        if 'block' in name:
            block_num = int(name.split('.')[1])
            if block_num >= args.grad_from_block:
                m.requires_grad = True
    return model

def build_backbone_clip(args):
    import clip

    # import text-img pre-trained CLIP backbone
    model, img_preprocess = clip.load("ViT-B/16")
    model = model.to(args.device)

    # freeze the entire Clip backbone
    for m in model.parameters():
        m.requires_grad = False

    return model

def build_direct_concat_model(args):
    # ----------------------
    # Backbone feature extraction creation: ViT-Dino or Clip
    # ----------------------
    if args.model_name == 'clip':
        feat_dim = 512
        model = build_backbone_clip(args)
    else:
        feat_dim = args.feat_dim
        model = build_backbone_vit(args)

    # ----------------------
    # Learned Single Heads: from previous steps
    # ----------------------
    single_heads_list = []
    for step in range(args.current_step):
        step_single = \
            LinearHeadPCC(in_dim=feat_dim, out_dim=args.num_novel_interval) \
            if args.normalization == 'pcc' \
            else LinearHead(in_dim=feat_dim, out_dim=args.num_novel_interval)

        step_single_state_dict = torch.load(args.learned_single_head_paths_list[step], map_location='cpu')
        step_single.load_state_dict(step_single_state_dict)
        single_heads_list.append(step_single)

    for s_single in single_heads_list:
        s_single.to(args.device)

    # ----------------------
    # To-be-learned Single Head for this step
    # ----------------------
    single_head = \
        LinearHeadPCC(in_dim=feat_dim, out_dim=args.num_novel_per_step) \
        if args.normalization == 'pcc' \
        else LinearHead(in_dim=feat_dim, out_dim=args.num_novel_per_step)

    single_head = single_head.to(args.device)

    # ----------------------
    # Joint Head Container
    # ----------------------
    joint_head = \
        LinearHeadPCC(in_dim=feat_dim, out_dim=args.current_novel_end) \
        if args.normalization == 'pcc' \
        else LinearHead(in_dim=feat_dim, out_dim=args.current_novel_end)

    joint_head = joint_head.to(args.device)
    return model, single_head, single_heads_list, joint_head
