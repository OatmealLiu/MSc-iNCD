import torch
from sklearn.cluster import KMeans

def build_kmeans(args):
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

    # ----------------------
    # Build K-Means models
    # ----------------------
    single_km_list = []
    joint_km_list = []

    # Fit step-wise single & joint KMeans
    for s in range(args.num_steps):
        if (1 + s) < args.num_steps:
            single_n_clusters = args.num_novel_interval
            joint_n_clusters = args.num_novel_interval * (1 + s)
        else:
            single_n_clusters = args.num_classes - args.num_novel_interval * s
            joint_n_clusters = args.num_classes

        print(f"\n------------> Single_cluster={single_n_clusters}, Joint_cluster={joint_n_clusters}")

        # single_km = KMeans(n_clusters=single_n_clusters, max_iter=args.km_max_iter, init='k-means++')   # 25 25 25 25
        # joint_km = KMeans(n_clusters=joint_n_clusters, max_iter=args.km_max_iter, init='k-means++')     # 25 50 75 100
        single_km = KMeans(n_clusters=single_n_clusters, max_iter=args.km_max_iter, init='random', n_init=1,
                           algorithm='full')   # 25 25 25 25
        joint_km = KMeans(n_clusters=joint_n_clusters, max_iter=args.km_max_iter, init='random',n_init=1,
                          algorithm='full')     # 25 50 75 100

        single_km_list.append(single_km)
        joint_km_list.append(joint_km)

    return model, single_km_list, joint_km_list