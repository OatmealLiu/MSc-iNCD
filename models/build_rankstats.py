import torch
import math
from functools import partial
from torch.nn import Parameter
import torch
import torch.nn as nn

from torch.nn.init import trunc_normal_
import torch.nn.functional as F

from .build_teacher_student_professor import build_backbone_vit

class FRoSTHead(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
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

class OcraHead(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.last_layer = NormedLinear(in_dim, out_dim)

    def forward(self, x):
        x = self.last_layer(x)
        return x

class NormedLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super(NormedLinear, self).__init__()
        self.weight = Parameter(torch.Tensor(in_features, out_features))
        self.weight.data.uniform_(-1, 1).renorm_(2, 1, 1e-5).mul_(1e5)

    def forward(self, x):
        out = F.normalize(x, dim=1).mm(F.normalize(self.weight, dim=0))
        return 10 * out

def build_rankstats(args):
    model = build_backbone_vit(args)

    # ----------------------
    # Teacher Heads: trained from previous steps
    # ----------------------
    teachers_list = []
    for step in range(args.current_step):
        if args.use_norm:
            teacher = OcraHead(in_dim=args.feat_dim, out_dim=args.num_novel_interval)
        else:
            teacher = FRoSTHead(in_dim=args.feat_dim, out_dim=args.num_novel_interval)

        teacher_state_dict = torch.load(args.pretrained_teacher_head_paths_list[step], map_location='cpu')
        teacher.load_state_dict(teacher_state_dict)
        print(f"Loaded trained teacher model from {args.pretrained_teacher_head_paths_list[step]}")
        teachers_list.append(teacher)

    for t_head in teachers_list:
        t_head.to(args.device)

    # ----------------------
    # Student Head: to be trained
    # ----------------------
    if args.use_norm:
        student = OcraHead(in_dim=args.feat_dim, out_dim=args.num_novel_per_step)
    else:
        student = FRoSTHead(in_dim=args.feat_dim, out_dim=args.num_novel_per_step)

    student = student.to(args.device)

    # ----------------------
    # Joint Head Container: container, no learning needed
    # ----------------------
    if args.use_norm:
        joint_head = OcraHead(in_dim=args.feat_dim, out_dim=args.current_novel_end)
    else:
        joint_head = FRoSTHead(in_dim=args.feat_dim, out_dim=args.current_novel_end)

    joint_head = joint_head.to(args.device)

    return model, teachers_list, student, joint_head

def build_eval_rankstats(args):
    model = build_backbone_vit(args)

    # ----------------------
    # Teacher Heads: trained from previous steps
    # ----------------------
    teachers_list = []
    for step in range(args.current_step):
        if args.use_norm:
            teacher = OcraHead(in_dim=args.feat_dim, out_dim=args.num_novel_interval)
        else:
            teacher = FRoSTHead(in_dim=args.feat_dim, out_dim=args.num_novel_interval)

        teacher_state_dict = torch.load(args.pretrained_teacher_head_paths_list[step], map_location='cpu')
        teacher.load_state_dict(teacher_state_dict)
        print(f"Loaded trained teacher model from {args.pretrained_teacher_head_paths_list[step]}")
        teachers_list.append(teacher)

    for t_head in teachers_list:
        t_head.to(args.device)

    # ----------------------
    # Student Head: to be trained
    # ----------------------
    if args.use_norm:
        student = OcraHead(in_dim=args.feat_dim, out_dim=args.num_novel_per_step)
    else:
        student = FRoSTHead(in_dim=args.feat_dim, out_dim=args.num_novel_per_step)

    student_state_dict = torch.load(args.save_student_path, map_location='cpu')
    student.load_state_dict(student_state_dict)
    student = student.to(args.device)

    # ----------------------
    # Joint Head Container: container, no learning needed
    # ----------------------
    if args.use_norm:
        joint_head = OcraHead(in_dim=args.feat_dim, out_dim=args.current_novel_end)
    else:
        joint_head = FRoSTHead(in_dim=args.feat_dim, out_dim=args.current_novel_end)

    # joint_state_dict = torch.load(args.save_joint_path, map_location='cpu')
    # joint_head.load_state_dict(joint_state_dict)
    joint_head = joint_head.to(args.device)

    return model, teachers_list, student, joint_head