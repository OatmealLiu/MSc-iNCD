import torch
import torch.nn as nn
from torch.nn.init import trunc_normal_

def weight_reinit(m):
    if isinstance(m, nn.Linear):
        trunc_normal_(m.weight, std=1.0)
        if isinstance(m, nn.Linear) and m.bias is not None:
            nn.init.constant_(m.bias, 0)

def build_scan(args):
    # backbone
    from .dino_backbones import build_ViT_B16_dino

    # classifier
    if args.use_norm:
        # our method uses both feature_norm and weight_norm (aka, CosNorm)
        from .build_scan_heads import HeadWnorm as Head
    else:
        # native SCAN does not use neither feature_norm nor weight_norm
        from .build_scan_heads import HeadWOnorm as Head

    # Build large-scale pre-trained backbone
    feat_dim = args.feat_dim
    encoder = build_ViT_B16_dino(args)

    # Prev-heads
    single_heads_dict = {
        'scan': [],
        'selflabel': []
    }
    for step in range(args.current_step):
        step_single_scan = Head(in_dim=feat_dim, out_dim=args.num_novel_interval)
        step_single_selflabel = Head(in_dim=feat_dim, out_dim=args.num_novel_interval)

        step_single_state_dict_scan = torch.load(args.learned_single_head_path_dict['scan'][step], map_location='cpu')
        step_single_state_dict_selflabel = torch.load(args.learned_single_head_path_dict['selflabel'][step],
                                                      map_location='cpu')

        step_single_scan.load_state_dict(step_single_state_dict_scan)
        step_single_selflabel.load_state_dict(step_single_state_dict_selflabel)

        step_single_scan = step_single_scan.to(args.device)
        step_single_selflabel = step_single_selflabel.to(args.device)
        single_heads_dict['scan'].append(step_single_scan)
        single_heads_dict['selflabel'].append(step_single_selflabel)


    # Current-head
    single_head = Head(in_dim=feat_dim, out_dim=args.num_novel_per_step)

    if args.dataset_name == 'cifar10' and args.use_norm is False:
        single_head.apply(weight_reinit)

    single_head = single_head.to(args.device)

    # Current-joint-head container
    joint_head_dict = {
        'scan': Head(in_dim=feat_dim, out_dim=args.current_novel_end),
        'selflabel': Head(in_dim=feat_dim, out_dim=args.current_novel_end)
    }
    joint_head_dict['scan'] = joint_head_dict['scan'].to(args.device)
    joint_head_dict['selflabel'] = joint_head_dict['selflabel'].to(args.device)

    return encoder, single_head, single_heads_dict, joint_head_dict
