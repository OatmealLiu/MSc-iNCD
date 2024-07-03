import torch

def build_ViT_B16_plain(args):
    from . import vision_transformer as vits
    model = vits.__dict__['vit_base']()
    model = model.to(args.device)
    return model

def build_uno(args):
    from .dino_backbones import build_ViT_B16_dino
    from .resnet_backbones import build_resnet18_plain, build_resnet18_imagenet1k, build_resnet50_plain, build_resnet50_dino
    from .linear_layer import LinearHeadCosNorm

    # ----------------------
    # Build the backbone
    # ----------------------
    if args.model_name == 'resnet18_plain':         # unlock
        feat_dim = 512
        model = build_resnet18_plain()
        model = model.to(args.device)
    elif args.model_name == 'resnet18_imagenet1k':
        feat_dim = 512
        model = build_resnet18_imagenet1k(args)
    elif args.model_name == 'resnet50_plain':       # unlock
        feat_dim = 2048
        model = build_resnet50_plain(args)
    elif args.model_name == 'resnet50_dino':        # unlock
        feat_dim = 2048
        model = build_resnet50_dino(args)
    elif args.model_name == 'vit_plain':            # unlock
        feat_dim = args.feat_dim
        model = build_ViT_B16_plain(args)
    elif args.model_name == 'vit_dino':             # unlock specified blocks
        feat_dim = args.feat_dim
        model = build_ViT_B16_dino(args)
    else:
        raise NotImplementedError

    # ----------------------
    # Linear Head for both base classes and novel classes, separately
    # ----------------------
    head_base = LinearHeadCosNorm(in_dim=feat_dim, out_dim=args.num_base)
    head_novel = LinearHeadCosNorm(in_dim=feat_dim, out_dim=args.num_novel)
    head_joint = LinearHeadCosNorm(in_dim=feat_dim, out_dim=args.num_classes)

    head_base = head_base.to(args.device)
    head_novel = head_novel.to(args.device)
    head_joint = head_joint.to(args.device)

    if args.device_count > 1:
        model = torch.nn.DataParallel(model).to(args.device)
        # head_base = torch.nn.DataParallel(model).to(args.device)
        # head_novel = torch.nn.DataParallel(model).to(args.device)
        # head_joint = torch.nn.DataParallel(model).to(args.device)

    return model, head_base, head_novel, head_joint
