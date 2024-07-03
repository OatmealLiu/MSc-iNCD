import torch
import torch.nn as nn


def load_dino_weights_from_cassle(path_ckpt):
    useless_keywords = {'projector', 'momentum', 'predictor', 'classifier', 'frozen', 'distill', 'prototypes', 'queue'}
    raw_ckpt_dict = torch.load(path_ckpt, map_location='cpu')
    raw_state_dict = raw_ckpt_dict["state_dict"]
    dino_state_dict = {}

    for k in raw_state_dict.keys():
        if any(key_ in k for key_ in useless_keywords):
            continue
        # if 'projector' in k or 'momentum' in k or 'predictor' in k or 'classifier' in k:
        #     continue
        dino_key = k.split("encoder.")[1]
        dino_state_dict[dino_key] = raw_state_dict[k]
    print(len(dino_state_dict.keys()))
    return dino_state_dict


# Miu: 这个是最重要的，每次都load cassle的weight
def build_backbone_vit(args):
    from . import vision_transformer as vits

    this_cassle_pretrain_ckpt_path = args.cassle_weight_path_dict[f"task{args.current_step}"]
    print(f"\n\n++++++++++++++++++>>>>>: {this_cassle_pretrain_ckpt_path}\n\n")

    # path_model_state_dict = args.dino_pretrain_path

    state_dict = load_dino_weights_from_cassle(this_cassle_pretrain_ckpt_path)

    model = vits.__dict__['vit_base']()
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

def build_direct_concat_model(args):
    # from . import vision_transformer as vits
    from .linear_layer import LinearHeadCosNorm

    feat_dim = args.feat_dim
    model = build_backbone_vit(args)

    # ----------------------
    # Learned Single Heads: from previous steps
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
    # To-be-learned Single Head for this step
    # ----------------------
    single_head = LinearHeadCosNorm(in_dim=feat_dim, out_dim=args.num_novel_per_step)

    single_head = single_head.to(args.device)

    # ----------------------
    # Joint Head Container
    # ----------------------
    joint_head = LinearHeadCosNorm(in_dim=feat_dim, out_dim=args.current_novel_end)

    joint_head = joint_head.to(args.device)
    return model, single_head, single_heads_list, joint_head
