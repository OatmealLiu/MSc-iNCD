import torch

def build_backbone_vit(args):
    from . import vision_transformer as vits
    path_model_state_dict = args.dino_pretrain_path

    model = vits.__dict__['vit_base']()
    state_dict = torch.load(path_model_state_dict, map_location='cpu')
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

def build_backbone_clip(args):
    import clip

    # import text-img pre-trained CLIP backbone
    model, img_preprocess = clip.load("ViT-B/16")
    model = model.to(args.device)

    # freeze the entire Clip backbone
    for m in model.parameters():
        m.requires_grad = False

    return model

def build_single_teacher_head(args):
    from . import vision_transformer as vits
    if args.model_name == 'clip':
        feat_dim = 512
    else:
        feat_dim = args.feat_dim

    if int(1+args.current_step) < args.num_steps:
        head = vits.__dict__['DINOHead'](
                            in_dim=feat_dim,
                            out_dim=args.num_novel_interval,
                            nlayers=args.num_mlp_layers) \
                            if args.model_head == 'DINOHead' \
                            else vits.__dict__['LinearHead'](
                            in_dim=feat_dim,
                            out_dim=args.num_novel_interval)
    else:
        head = vits.__dict__['DINOHead'](
                            in_dim=feat_dim,
                            out_dim=args.num_novel_per_step,
                            nlayers=args.num_mlp_layers) \
                            if args.model_head == 'DINOHead' \
                            else vits.__dict__['LinearHead'](
                            in_dim=feat_dim,
                            out_dim=args.num_novel_per_step)

    head = head.to(args.device)
    return head

def build_step_teacher_heads(args):
    from . import vision_transformer as vits
    if args.model_name == 'clip':
        feat_dim = 512
    else:
        feat_dim = args.feat_dim

    # ----------------------
    # Teacher Heads
    # ----------------------
    teachers_list = []
    for step in range(1+args.current_step):
        if int(1 + step) < args.num_steps:
            teacher = vits.__dict__['DINOHead'](
                                in_dim=feat_dim,
                                out_dim=args.num_novel_interval,
                                nlayers=args.num_mlp_layers) \
                                if args.model_head == 'DINOHead' \
                                else vits.__dict__['LinearHead'](
                                in_dim=feat_dim,
                                out_dim=args.num_novel_interval)
        else:
            teacher = vits.__dict__['DINOHead'](
                                in_dim=feat_dim,
                                out_dim=args.num_novel_per_step,
                                nlayers=args.num_mlp_layers) \
                                if args.model_head == 'DINOHead' \
                                else vits.__dict__['LinearHead'](
                                in_dim=feat_dim,
                                out_dim=args.num_novel_per_step)

        teacher_state_dict = torch.load(args.pretrained_teacher_head_paths_list[step], map_location='cpu')
        teacher.load_state_dict(teacher_state_dict)
        teachers_list.append(teacher)

    for t_head in teachers_list:
        t_head.to(args.device)
    return teachers_list


def build_teacher(args):
    if args.model_name == 'clip':
        feat_dim = 512
        model = build_backbone_clip(args)
    else:
        feat_dim = args.feat_dim
        model = build_backbone_vit(args)

    teacher = build_single_teacher_head(args)
    return model, teacher

def build_teacher_student(args):
    from . import vision_transformer as vits
    if args.model_name == 'clip':
        feat_dim = 512
        model = build_backbone_clip(args)
    else:
        feat_dim = args.feat_dim
        model = build_backbone_vit(args)

    teachers_list = build_step_teacher_heads(args)

    # ----------------------
    # Student Head
    # ----------------------
    student = vits.__dict__['DINOHead'](
                        in_dim=feat_dim,
                        out_dim=args.current_novel_end,
                        nlayers=args.num_mlp_layers) \
                        if args.model_head == 'DINOHead' \
                        else vits.__dict__['LinearHead'](
                        in_dim=feat_dim,
                        out_dim=args.current_novel_end)

    student = student.to(args.device)

    for step in range(1+args.current_step):
        w_saved = teachers_list[step].last_layer.weight.data.clone()
        student.last_layer.weight.data[step*args.num_novel_interval:(1+step)*args.num_novel_interval] = w_saved

    return model, teachers_list, student

def build_teacher_student_professor(args):
    from . import vision_transformer as vits
    model, teacher_list, student = build_teacher_student(args)

    # ----------------------
    # Professor Heads
    # ----------------------
    professors_list = []
    for step in range(len(teacher_list)):
        professor = vits.__dict__['DINOHead'](
                                in_dim=args.feat_dim,
                                out_dim=args.current_novel_end,
                                nlayers=args.num_mlp_layers) \
                                if args.model_head == 'DINOHead' \
                                else vits.__dict__['LinearHead'](
                                in_dim=args.feat_dim,
                                out_dim=args.current_novel_end)
        professor = professor.to(args.device)
        w_saved = teacher_list[step].last_layer.weight.data.clone()
        zero_pad_weights = torch.zeros(professor.last_layer.weight.data.shape).to(args.device)
        zero_pad_weights[step * args.num_novel_interval:(1 + step) * args.num_novel_interval] = w_saved
        professor.last_layer.weight.data = zero_pad_weights
        professors_list.append(professor)

    return model, teacher_list, professors_list, student

def build_student_solo(args):
    from . import vision_transformer as vits
    model = build_backbone_vit(args)

    # ----------------------
    # Teacher Heads: from previous steps
    # ----------------------
    teachers_list = []
    for step in range(args.current_step):
        teacher = vits.__dict__['DINOHead'](
                            in_dim=args.feat_dim,
                            out_dim=args.num_novel_interval,
                            nlayers=args.num_mlp_layers) \
                            if args.model_head == 'DINOHead' \
                            else vits.__dict__['LinearHead'](
                            in_dim=args.feat_dim,
                            out_dim=args.num_novel_interval)

        teacher_state_dict = torch.load(args.pretrained_teacher_head_paths_list[step], map_location='cpu')
        teacher.load_state_dict(teacher_state_dict)
        teachers_list.append(teacher)

    for t_head in teachers_list:
        t_head.to(args.device)

    # ----------------------
    # Student Head
    # ----------------------
    student = vits.__dict__['DINOHead'](
        in_dim=args.feat_dim,
        out_dim=args.num_novel_per_step,
        nlayers=args.num_mlp_layers) \
        if args.model_head == 'DINOHead' \
        else vits.__dict__['LinearHead'](
        in_dim=args.feat_dim,
        out_dim=args.num_novel_per_step)

    student = student.to(args.device)

    # ----------------------
    # Joint Head Container
    # ----------------------
    joint_head = vits.__dict__['DINOHead'](
        in_dim=args.feat_dim,
        out_dim=args.current_novel_end,
        nlayers=args.num_mlp_layers) \
        if args.model_head == 'DINOHead' \
        else vits.__dict__['LinearHead'](
        in_dim=args.feat_dim,
        out_dim=args.current_novel_end)

    joint_head = joint_head.to(args.device)
    return model, teachers_list, student, joint_head


def build_weight_discrepancy(args):
    from . import vision_transformer as vits
    model = build_backbone_vit(args)

    # ----------------------
    # Teacher Heads: from previous steps
    # ----------------------
    teachers_list = []
    for step in range(args.current_step):
        teacher = vits.__dict__['DINOHead'](
            in_dim=args.feat_dim,
            out_dim=args.num_novel_interval,
            nlayers=args.num_mlp_layers) \
            if args.model_head == 'DINOHead' \
            else vits.__dict__['LinearHead'](
            in_dim=args.feat_dim,
            out_dim=args.num_novel_interval)

        teacher_state_dict = torch.load(args.pretrained_teacher_head_paths_list[step], map_location='cpu')
        teacher.load_state_dict(teacher_state_dict)
        teachers_list.append(teacher)

    for t_head in teachers_list:
        t_head.to(args.device)

    # ----------------------
    # Student Head
    # ----------------------
    student = vits.__dict__['DINOHead'](
        in_dim=args.feat_dim,
        out_dim=args.num_novel_per_step,
        nlayers=args.num_mlp_layers) \
        if args.model_head == 'DINOHead' \
        else vits.__dict__['LinearHead'](
        in_dim=args.feat_dim,
        out_dim=args.num_novel_per_step)

    if args.warmup:
        warmed_student_head_state_dict = torch.load(args.warmup_student_head_path, map_location='cpu')
        student.load_state_dict(warmed_student_head_state_dict)
        print(f"Loaded warmed-up student head weights from {args.warmup_student_head_path}")

    student = student.to(args.device)

    # ----------------------
    # Joint Head Container
    # ----------------------
    joint_head = vits.__dict__['DINOHead'](
        in_dim=args.feat_dim,
        out_dim=args.current_novel_end,
        nlayers=args.num_mlp_layers) \
        if args.model_head == 'DINOHead' \
        else vits.__dict__['LinearHead'](
        in_dim=args.feat_dim,
        out_dim=args.current_novel_end)

    joint_head = joint_head.to(args.device)
    return model, teachers_list, student, joint_head
