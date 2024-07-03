import torch

def build_data_fading(args):
    from .linear_layer import LinearHead, LinearHeadCosNorm, LinearHeadScaledCosNorm
    from .dino_backbones import build_ViT_B16_dino
    from .clip_backbones import build_ViT_B16_clip

    if args.model_name == 'clip':
        feat_dim = 512
        model = build_ViT_B16_clip(args)
    else:
        feat_dim = args.feat_dim
        model = build_ViT_B16_dino(args)

    # ----------------------
    # Teacher Heads: from previous steps
    # ----------------------
    single_heads_list = []
    for step in range(args.current_step):
        step_single = LinearHeadCosNorm(in_dim=feat_dim, out_dim=args.num_novel_interval)
        step_single_state_dict = torch.load(args.learned_single_head_paths_list[step], map_location='cpu')
        step_single.load_state_dict(step_single_state_dict)
        single_heads_list.append(step_single)

    for s_single in single_heads_list:
        s_single.to(args.device)

    # ----------------------
    # Student Head
    # ----------------------
    single_head = LinearHeadCosNorm(in_dim=feat_dim, out_dim=args.num_novel_per_step)
    single_head = single_head.to(args.device)

    # ----------------------
    # Joint Head Container
    # ----------------------
    joint_head = LinearHeadCosNorm(in_dim=feat_dim, out_dim=args.current_novel_end)
    joint_head = joint_head.to(args.device)

    return model, single_head, single_heads_list, joint_head
