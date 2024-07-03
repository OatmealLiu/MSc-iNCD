import torch

def build_manifold_hopper(args):
    if args.feat_slice == 'rank':
        from .linear_layer import LinearHeadMultiManifoldCosNormRank as LinearHead
    else:
        from .linear_layer import LinearHeadMultiManifoldCosNorm as LinearHead

    from .dino_backbones import build_ViT_B16_dino

    feat_dim = args.feat_dim
    model = build_ViT_B16_dino(args)
    manifolds = [feat_dim, feat_dim-args.dim_reduction, feat_dim-2*args.dim_reduction]

    # ----------------------
    # Teacher Heads: from previous steps
    # ----------------------
    single_heads_list = []
    for step in range(args.current_step):
        step_single = LinearHead(manifolds=manifolds, out_dim=args.num_novel_interval)
        step_single_state_dict = torch.load(args.learned_single_head_paths_list[step], map_location='cpu')
        step_single.load_state_dict(step_single_state_dict)
        single_heads_list.append(step_single)

    for s_single in single_heads_list:
        s_single.to(args.device)

    # ----------------------
    # Student Head
    # ----------------------
    single_head = LinearHead(manifolds=manifolds, out_dim=args.num_novel_per_step)
    single_head = single_head.to(args.device)

    # ----------------------
    # Joint Head Container
    # ----------------------
    joint_head = LinearHead(manifolds=manifolds, out_dim=args.current_novel_end)
    joint_head = joint_head.to(args.device)

    return model, single_head, single_heads_list, joint_head
