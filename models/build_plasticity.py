import torch
from . import vision_transformer as vits

def build_backbone_vit(args):
    path_model_state_dict = args.load_model_path

    model = vits.__dict__['vit_base']()
    state_dict = torch.load(path_model_state_dict, map_location='cpu')
    model.load_state_dict(state_dict)
    model = model.to(args.device)

    for m in model.parameters():
        m.requires_grad = False

    # Only finetune layers from block 'args.grad_from_block' onwards
    for name, m in model.named_parameters():
        if 'block' in name:
            block_num = int(name.split('.')[1])
            if block_num >= args.grad_from_block:
                m.requires_grad = True
    return model

def build_plasticity(args):
    # To-be-learned model and head for current step
    model = build_backbone_vit(args)
    single_head = vits.__dict__['LinearHead'](in_dim=args.feat_dim, out_dim=args.num_novel_per_step)
    single_head = single_head.to(args.device)

    # Learned model-head pairs in prev-steps
    learned_model_head_pair_list = []
    for step in range(args.current_step):
        # model
        this_model = vits.__dict__['vit_base']()

        if args.device_count > 1:
            this_model = torch.nn.DataParallel(this_model).to(args.device)

        this_model_state_dict = torch.load(args.learned_model_paths_list[step])
        this_model.load_state_dict(this_model_state_dict)

        if args.device_count < 1:
            this_model = this_model.to(args.device)

        for m in this_model.parameters():
            m.requires_grad = False

        # head
        this_head = vits.__dict__['LinearHead'](in_dim=args.feat_dim, out_dim=args.num_novel_interval)

        this_head_state_dict = torch.load(args.learned_single_head_paths_list[step], map_location='cpu')
        this_head.load_state_dict(this_head_state_dict)
        this_head = this_head.to(args.device)

        learned_model_head_pair_list.append((this_model, this_head))

    # Joint head container
    joint_head = vits.__dict__['LinearHead'](in_dim=args.feat_dim, out_dim=args.current_novel_end)
    joint_head = joint_head.to(args.device)

    if args.device_count > 1:
        model = torch.nn.DataParallel(model).to(args.device)

    return model, single_head, learned_model_head_pair_list, joint_head

