import torch


def build_ewc_model(args):
    from . import vision_transformer as vits
    from .linear_layer import LinearHeadCosNorm as Head

    # ----------------------
    # Encoder_shared (init with 100% frozen)
    # ----------------------
    feat_dim = args.feat_dim
    path_model_state_dict = args.dino_pretrain_path

    encoder = vits.__dict__['vit_base']()
    state_dict = torch.load(path_model_state_dict, map_location='cpu')
    encoder.load_state_dict(state_dict)
    encoder = encoder.to(args.device)

    for m in encoder.parameters():
        m.requires_grad = False

    # ----------------------
    # Step-wise: single_head, joint_head
    # ----------------------
    step_single_head_list = []
    joint_head_container_list = []
    for s in range(args.num_steps):
        # Single Head
        single_head = Head(in_dim=feat_dim, out_dim=args.num_novel_interval)
        single_head = single_head.to(args.device)
        step_single_head_list.append(single_head)
        # Joint Head
        joint_head = Head(in_dim=feat_dim, out_dim=(1+s)*args.num_novel_interval)
        joint_head = joint_head.to(args.device)
        joint_head_container_list.append(joint_head)

    return encoder, step_single_head_list, joint_head_container_list
