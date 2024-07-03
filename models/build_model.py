import torch
import torch.nn as nn


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

def build_resnet50_dino(args):
    model = torch.hub.load('facebookresearch/dino:main', 'dino_resnet50')
    model = model.to(args.device)

    for m in model.parameters():
        m.requires_grad = False
    return model

def build_resnet18_imagenet1k(args):
    from torchvision.models.resnet import resnet18
    model = resnet18(pretrained=True)
    model.fc = nn.Identity()
    if 'cifar' in args.dataset_name:
        model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=2, bias=False)
        model.maxpool = nn.Identity()

    model = model.to(args.device)

    for m in model.parameters():
        m.requires_grad = False
    return model

def build_direct_concat_model(args):
    # from . import vision_transformer as vits
    from .linear_layer import LinearHeadCosNorm

    if args.model_name == 'clip':
        feat_dim = 512
        model = build_backbone_clip(args)
    elif args.model_name == 'resnet18_imagenet1k':
        feat_dim = 512
        model = build_resnet18_imagenet1k(args)
    elif args.model_name == 'resnet50_dino':
        feat_dim = 2048
        model = build_resnet50_dino(args)
    else:
        feat_dim = args.feat_dim
        model = build_backbone_vit(args)

    # ----------------------
    # Learned Single Heads: from previous steps
    # ----------------------
    single_heads_list = []
    for step in range(args.current_step):
        # step_single = vits.__dict__['DINOHead'](
        #                     in_dim=feat_dim,
        #                     out_dim=args.num_novel_interval,
        #                     nlayers=args.num_mlp_layers) \
        #                     if args.model_head == 'DINOHead' \
        #                     else vits.__dict__['LinearHead'](
        #                     in_dim=feat_dim,
        #                     out_dim=args.num_novel_interval)
        step_single = LinearHeadCosNorm(in_dim=feat_dim, out_dim=args.num_novel_interval)

        step_single_state_dict = torch.load(args.learned_single_head_paths_list[step], map_location='cpu')
        step_single.load_state_dict(step_single_state_dict)
        single_heads_list.append(step_single)

    for s_single in single_heads_list:
        s_single.to(args.device)

    # ----------------------
    # To-be-learned Single Head for this step
    # ----------------------
    # single_head = vits.__dict__['DINOHead'](
    #     in_dim=feat_dim,
    #     out_dim=args.num_novel_per_step,
    #     nlayers=args.num_mlp_layers) \
    #     if args.model_head == 'DINOHead' \
    #     else vits.__dict__['LinearHead'](
    #     in_dim=feat_dim,
    #     out_dim=args.num_novel_per_step)
    single_head = LinearHeadCosNorm(in_dim=feat_dim, out_dim=args.num_novel_per_step)

    single_head = single_head.to(args.device)

    # ----------------------
    # Joint Head Container
    # ----------------------
    # joint_head = vits.__dict__['DINOHead'](
    #     in_dim=feat_dim,
    #     out_dim=args.current_novel_end,
    #     nlayers=args.num_mlp_layers) \
    #     if args.model_head == 'DINOHead' \
    #     else vits.__dict__['LinearHead'](
    #     in_dim=feat_dim,
    #     out_dim=args.current_novel_end)
    joint_head = LinearHeadCosNorm(in_dim=feat_dim, out_dim=args.current_novel_end)

    joint_head = joint_head.to(args.device)
    return model, single_head, single_heads_list, joint_head
