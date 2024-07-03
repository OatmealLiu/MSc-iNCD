import torch
import math
from functools import partial
from torch.nn import Parameter
import torch
import torch.nn as nn

from torch.nn.init import trunc_normal_
import torch.nn.functional as F

from .build_teacher_student_professor import build_backbone_vit

# class OcraHeadInPlace(nn.Module):
#     def __init__(self, in_dim, out_dim, use_norm):
#         super().__init__()
#         self.use_norm = use_norm
#         self.last_layer = nn.Linear(in_dim, out_dim, bias=False)
#         self.apply(self._init_weights)
#
#     def _init_weights(self, m):
#         if isinstance(m, nn.Linear):
#             trunc_normal_(m.weight, std=.02)
#             if isinstance(m, nn.Linear) and m.bias is not None:
#                 nn.init.constant_(m.bias, 0)
#
#     def forward(self, x):
#         if self.use_norm:
#             x = nn.functional.normalize(x, dim=-1, p=2)
#         x = self.last_layer(x)
#
#         return x

class OcraHead(nn.Module):
    def __init__(self, in_dim, out_dim, use_norm, feat_norm, T=1.0):
        super().__init__()
        self.use_norm = use_norm
        self.feat_norm = feat_norm
        self.T = T
        if use_norm:
            self.last_layer = NormedLinear(in_dim, out_dim)
        else:
            self.last_layer = nn.Linear(in_dim, out_dim, bias=False)
            self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=2.1)    # OG=.02
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        if self.use_norm:
            x = self.last_layer(x)
        else:
            # try 1
            if self.feat_norm:
                x = nn.functional.normalize(x, dim=-1, p=2)
                x = self.last_layer(x)
            else:
                x = self.last_layer(x) / self.T      # /10 is the trick
        return x

class NormedLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super(NormedLinear, self).__init__()
        self.weight = Parameter(torch.Tensor(in_features, out_features))
        self.weight.data.uniform_(-1, 1).renorm_(2, 1, 1e-5).mul_(1e5)

    def forward(self, x):
        out = F.normalize(x, dim=1).mm(F.normalize(self.weight, dim=0))
        return 10 * out

def build_orca(args):
    model = build_backbone_vit(args)

    # ----------------------
    # Teacher Heads: trained from previous steps
    # ----------------------
    teachers_list = []
    for step in range(args.current_step):
        teacher = OcraHead(in_dim=args.feat_dim, out_dim=args.num_novel_interval, use_norm=args.use_norm,
                           feat_norm=args.feat_norm, T=args.softmax_temp)

        teacher_state_dict = torch.load(args.pretrained_teacher_head_paths_list[step], map_location='cpu')
        teacher.load_state_dict(teacher_state_dict)
        print(f"Loaded trained teacher model from {args.pretrained_teacher_head_paths_list[step]}")
        teachers_list.append(teacher)

    for t_head in teachers_list:
        t_head.to(args.device)

    # ----------------------
    # Student Head: to be trained
    # ----------------------
    student = OcraHead(in_dim=args.feat_dim, out_dim=args.num_novel_per_step, use_norm=args.use_norm,
                       feat_norm=args.feat_norm, T=args.softmax_temp)

    student = student.to(args.device)

    # ----------------------
    # Joint Head Container: container, no learning needed
    # ----------------------
    joint_head = OcraHead(in_dim=args.feat_dim, out_dim=args.current_novel_end, use_norm=args.use_norm,
                          feat_norm=args.feat_norm, T=args.softmax_temp)

    joint_head = joint_head.to(args.device)

    return model, teachers_list, student, joint_head

