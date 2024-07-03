import torch
import torch.nn as nn
from torch.nn.init import trunc_normal_

from .build_teacher_student_professor import build_backbone_vit
from .vit_addson import vit_base_relu

class LinearHead(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        # self.apply(self._init_weights)
        self.last_layer = nn.Linear(in_dim, out_dim, bias=False)

    def forward(self, x):
        x = nn.functional.normalize(x, dim=-1)
        x = self.last_layer(x)
        return x/0.1

    @torch.no_grad()
    def normalize_prototypes(self):
        w = self.last_layer.weight.data.clone()
        w = torch.nn.functional.normalize(w, dim=1, p=2)
        self.last_layer.weight.copy_(w)

    # def _init_weights(self, m):
    #     if isinstance(m, nn.Linear):
    #         trunc_normal_(m.weight, std=.02)
    #         if isinstance(m, nn.Linear) and m.bias is not None:
    #             nn.init.constant_(m.bias, 0)

# class LinearHead(nn.Module):
#     def __init__(self, in_dim, out_dim, norm_last_layer=True):
#         super().__init__()
#         self.apply(self._init_weights)
#         self.last_layer = nn.utils.weight_norm(nn.Linear(in_dim, out_dim, bias=False))
#         self.last_layer.weight_g.data.fill_(1)
#         if norm_last_layer:
#             self.last_layer.weight_g.requires_grad = False
#
#     def _init_weights(self, m):
#         if isinstance(m, nn.Linear):
#             trunc_normal_(m.weight, std=.02)
#             if isinstance(m, nn.Linear) and m.bias is not None:
#                 nn.init.constant_(m.bias, 0)
#
#     def forward(self, x):
#         x = nn.functional.normalize(x, dim=-1, p=2)
#         x = self.last_layer(x)
#         return x

# class LinearHead(nn.Module):
#     def __init__(self, in_dim, out_dim, init_std=.2):
#         super().__init__()
#         self.init_std = init_std
#         self.last_layer = nn.Linear(in_dim, out_dim, bias=False)
#         self.apply(self._init_weights)
#
#     def _init_weights(self, m):
#         if isinstance(m, nn.Linear):
#             trunc_normal_(m.weight, std=.2)  #og: .02
#             if isinstance(m, nn.Linear) and m.bias is not None:
#                 nn.init.constant_(m.bias, 0)
#
#     def forward(self, x):
#         x = self.last_layer(x)
#         return x

def build_frost_model(args):
    from . import vision_transformer as vits
    from .linear_layer import LinearHeadCosNorm

    if args.current_step == 0:
        # ----------------------
        # Build single task-specific model+head for this current NCD step
        # ----------------------
        model = build_backbone_vit(args)
        single_head = LinearHeadCosNorm(in_dim=args.feat_dim, out_dim=args.num_novel_per_step)
        single_head = single_head.to(args.device)

        return model, single_head

    # ----------------------
    # Build single task-specific model+head for this current NCD step
    # ----------------------
    model = vits.__dict__['vit_base']()
    state_dict = torch.load(args.prev_single_backbone_paths_list[-1], map_location='cpu') # start from nearest model
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

    single_head = LinearHeadCosNorm(in_dim=args.feat_dim, out_dim=args.num_novel_per_step)
    single_head = single_head.to(args.device)

    # ----------------------
    # Load single task-specific model+head for this NCD step
    # ----------------------
    prev_pair_list = []
    for step in range(args.current_step):
        # model
        prev_model = vits.__dict__['vit_base']()
        prev_model_state_dict = torch.load(args.prev_single_backbone_paths_list[step], map_location='cpu')
        prev_model.load_state_dict(prev_model_state_dict)
        prev_model = prev_model.to(args.device)

        for m in prev_model.parameters():
            m.requires_grad = False

        # head
        prev_head = LinearHeadCosNorm(in_dim=args.feat_dim, out_dim=args.num_novel_interval)
        prev_head_state_dict = torch.load(args.prev_single_head_paths_list[step], map_location='cpu')
        prev_head.load_state_dict(prev_head_state_dict)
        prev_head = prev_head.to(args.device)

        prev_pair_list.append((prev_model, prev_head))

    # ----------------------
    # Build joint task-agnostic model+head for this NCD step, to be initialized after student training
    # ----------------------
    #       head
    joint_head = LinearHeadCosNorm(in_dim=args.feat_dim, out_dim=args.current_novel_end)
    joint_head = joint_head.to(args.device)

    for step in range(args.current_step):
        w_saved = prev_pair_list[step][1].last_layer.weight.data.clone()
        joint_head.last_layer.weight.data[step * args.num_novel_interval:(1 + step) * args.num_novel_interval]\
            .copy_(w_saved)

    current_w_saved = single_head.last_layer.weight.data.clone()
    joint_head.last_layer.weight.data[args.current_novel_start:args.current_novel_end].copy_(current_w_saved)

    # for the first (init) step_teacher_pair_list is a empty list
    return model, single_head, joint_head, prev_pair_list

