clip_available_models = ['RN50', 'RN101', 'RN50x4', 'RN50x16', 'ViT-B/32', 'ViT-B/16']

def build_resnet50_clip(args):
    import clip
    model, img_preprocess = clip.load(clip_available_models[0])
    model = model.to(args.device)
    for m in model.parameters():
        m.requires_grad = False
    return model

def build_resnet101_clip(args):
    import clip
    model, img_preprocess = clip.load(clip_available_models[1])
    model = model.to(args.device)
    for m in model.parameters():
        m.requires_grad = False
    return model


def build_resnet50x4_clip(args):
    import clip
    model, img_preprocess = clip.load(clip_available_models[2])
    model = model.to(args.device)
    for m in model.parameters():
        m.requires_grad = False
    return model


def build_resnet50x16_clip(args):
    import clip
    model, img_preprocess = clip.load(clip_available_models[3])
    model = model.to(args.device)
    for m in model.parameters():
        m.requires_grad = False
    return model


def build_ViT_B32_clip(args):
    import clip
    model, img_preprocess = clip.load(clip_available_models[4])
    model = model.to(args.device)
    for m in model.parameters():
        m.requires_grad = False
    return model


def build_ViT_B16_clip(args):
    import clip
    model, img_preprocess = clip.load(clip_available_models[5])
    model = model.to(args.device)
    for m in model.parameters():
        m.requires_grad = False
    return model

