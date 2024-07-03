import torch
from . import vision_transformer as vits
from .build_teacher_student_professor import build_backbone_vit


def build_unlocked_model(args):
    if args.stage == 'stage1':
        # ----------------------
        # Build single task-specific model+head for this NCD step
        # ----------------------
        student_model = build_backbone_vit(args)

        student_head = vits.__dict__['DINOHead'](
                                in_dim=args.feat_dim,
                                out_dim=args.num_novel_per_step,
                                nlayers=args.num_mlp_layers) \
                                if args.model_head == 'DINOHead' \
                                else vits.__dict__['LinearHead'](
                                in_dim=args.feat_dim,
                                out_dim=args.num_novel_per_step)

        warmed_student_head_state_dict = torch.load(args.warmup_student_head_path, map_location='cpu')
        student_head.load_state_dict(warmed_student_head_state_dict)
        print(f"Loaded warmed-up student head weights from {args.warmup_student_head_path}")
        student_head.to(args.device)

        return student_model, student_head

    # else case: args.stage == 'stage2' or eval mode
    # ----------------------
    # Load single task-specific model+head for this NCD step
    # ----------------------
    teacher_pair_list = []
    for step in range(1+args.current_step):
        # model
        teacher_model = vits.__dict__['vit_base']()
        teacher_model_state_dict = torch.load(args.pretrained_teacher_backbone_paths_list[step], map_location='cpu')
        teacher_model.load_state_dict(teacher_model_state_dict)
        teacher_model = teacher_model.to(args.device)

        for m in teacher_model.parameters():
            m.requires_grad = False

        # Only finetune layers from block 'args.grad_from_block' onwards
        for name, m in teacher_model.named_parameters():
            if 'block' in name:
                block_num = int(name.split('.')[1])
                if block_num >= args.grad_from_block:
                    m.requires_grad = True

        # head
        if int(1 + step) < args.num_steps:
            teacher_head = vits.__dict__['DINOHead'](
                in_dim=args.feat_dim,
                out_dim=args.num_novel_interval,
                nlayers=args.num_mlp_layers) \
                if args.model_head == 'DINOHead' \
                else vits.__dict__['LinearHead'](
                in_dim=args.feat_dim,
                out_dim=args.num_novel_interval)
        else:
            teacher_head = vits.__dict__['DINOHead'](
                in_dim=args.feat_dim,
                out_dim=args.num_novel_interval,
                nlayers=args.num_mlp_layers) \
                if args.model_head == 'DINOHead' \
                else vits.__dict__['LinearHead'](
                in_dim=args.feat_dim,
                out_dim=args.num_novel_per_step)

        teacher_head_state_dict = torch.load(args.pretrained_teacher_head_paths_list[step], map_location='cpu')
        teacher_head.load_state_dict(teacher_head_state_dict)
        teacher_head = teacher_head.to(args.device)

        teacher_pair_list.append((teacher_model, teacher_head))

    # ----------------------
    # Build joint task-agnostic model+head for this NCD step, to be initialized after student training
    # ----------------------
    #       model
    joint_model = build_backbone_vit(args)

    #       head
    joint_head = vits.__dict__['DINOHead'](
        in_dim=args.feat_dim,
        out_dim=args.current_novel_end,
        nlayers=args.num_mlp_layers) \
        if args.model_head == 'DINOHead' \
        else vits.__dict__['LinearHead'](
        in_dim=args.feat_dim,
        out_dim=args.current_novel_end)

    joint_head = joint_head.to(args.device)

    for step in range(args.current_step):
        w_saved = teacher_pair_list[step][1].last_layer.weight.data.clone()
        joint_head.last_layer.weight.data[step * args.num_novel_interval:(1 + step) * args.num_novel_interval]\
            .copy_(w_saved)

    current_w_saved = teacher_pair_list[args.current_step][1].last_layer.weight.data.clone()
    joint_head.last_layer.weight.data[args.current_novel_start:args.current_novel_end].copy_(current_w_saved)
    # for the first (init) step_teacher_pair_list is a empty list
    return teacher_pair_list, joint_model, joint_head

