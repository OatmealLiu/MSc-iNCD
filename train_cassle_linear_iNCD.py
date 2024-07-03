import torch
from utils.util import seed_torch
from utils.logging import Logger
import os
import sys
import copy
import wandb
import math

from utils.sinkhorn_knopp import SinkhornKnopp

from data.build_dataset import build_data
from data.config_dataset import set_dataset_config_cluster

from models.build_cassle import build_direct_concat_model
from methods.cassle import DirectConcat

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='cluster', formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # CaSSLe related
    parser.add_argument('--ucl_method', type=str, default='swav', choices=['swav', 'byol'])

    # Hyper-parameters Setting
    parser.add_argument('--epochs', default=200, type=int)
    parser.add_argument('--batch_size', default=128, type=int)

    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--gamma', type=float, default=0.1)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight_decay', type=float, default=5e-5)

    parser.add_argument('--apply_l2weights', action='store_true', default=False,
                        help='L2 norm classifier weights or not, for control experiments')

    # UNO knobs
    parser.add_argument("--softmax_temp", default=0.1, type=float, help="softmax temperature")
    parser.add_argument("--threshold", default=0.5, type=float, help="threshold for hard pseudo-labeling")
    parser.add_argument("--num_iters_sk", default=3, type=int, help="number of iters for Sinkhorn")
    parser.add_argument("--epsilon_sk", default=0.05, type=float, help="epsilon for the Sinkhorn")
    parser.add_argument('--alpha', default=0.75, type=float)

    # Dataset Setting
    parser.add_argument('--dataset_name', type=str, default='cifar100', choices=['cifar10', 'cifar100', 'tinyimagenet',
                                                                                 'cub200', 'herb19', 'scars',
                                                                                 'aircraft'])
    # parser.add_argument('--dataset_root', type=str, default='./data/datasets/CIFAR/')
    # parser.add_argument('--num_classes', default=100, type=int)

    parser.add_argument('--aug_type', type=str, default='vit_uno', choices=['vit_frost', 'vit_uno', 'resnet',
                                                                            'vit_uno_clip'])
    parser.add_argument('--num_workers', default=8, type=int)

    # Strategy Setting
    parser.add_argument('--num_steps', default=10, type=int)
    parser.add_argument('--current_step', default=0, type=int)

    # Model Config
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'eval'])
    parser.add_argument('--model_name', type=str, default='vit_dino', choices=['vit_dino', 'clip', 'resnet50_dino',
                                                                               'resnet18_imagenet1k'])
    parser.add_argument('--grad_from_block', type=int, default=12)  # 12->do not fine tune backbone at all
    parser.add_argument('--num_mlp_layers', type=int, default=1)  # 12->do not fine tune backbone at all
    parser.add_argument('--dino_pretrain_path', type=str,
                        default='./models/dino_weights/dino_vitbase16_pretrain.pth')
    parser.add_argument('--model_head', type=str, default='LinearHead', choices=['LinearHead', 'DINOHead'])

    # Experimental Setting
    parser.add_argument('--seed', default=10, type=int)

    parser.add_argument('--exp_root', type=str, default='./outputs_cassle/')
    parser.add_argument('--weights_root', type=str, default='./models/single_weights_cassle/')

    parser.add_argument('--exp_marker', type=str, default='nonsense_expt')
    parser.add_argument('--wandb_mode', type=str, default='online', choices=['online', 'offline', 'disabled'])
    parser.add_argument('--wandb_entity', type=str, default='oatmealliu')

    # ----------------------
    # Initial Configurations
    # ----------------------
    args = parser.parse_args()

    # init. dataset config.
    args = set_dataset_config_cluster(args)

    # init. config.
    args.cuda = torch.cuda.is_available()
    device = torch.device("cuda" if args.cuda else "cpu")
    args.device = torch.device("cuda" if args.cuda else "cpu")
    seed_torch(args.seed)

    # init. experimental output path
    runner_name = os.path.basename(__file__).split(".")[0]

    # Experimental Dir.
    model_dir = os.path.join(args.exp_root, f"{runner_name}_{args.ucl_method}_{args.dataset_name}_Steps{args.num_steps}")
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    # Single head weights Dir.
    weights_dir = os.path.join(args.weights_root, f"{args.ucl_method}_{args.dataset_name}_Steps{args.num_steps}")
    if not os.path.exists(weights_dir):
        os.makedirs(weights_dir)

    # Miu: CaSSLe pre-trained weights
    # ./models/cassle_weights/swav/cifar10_2steps
    args.cassle_weight_path_dict = {}
    cassle_weight_dir = f"../global_weights/cassle_weights/{args.ucl_method}/{args.dataset_name}_{args.num_steps}steps/"
    weights_fnames = os.listdir(cassle_weight_dir)
    assert len(weights_fnames) == args.num_steps
    for task_idx in range(args.num_steps):#0-4 or 0-1
        task_name = f"task{task_idx}"
        for fname in weights_fnames:
            if task_name in fname:
                args.cassle_weight_path_dict[task_name] = cassle_weight_dir + fname

    # path to pre-trained single heads weights .pth file
    args.learned_single_head_paths_list = []
    for step in range(args.current_step):
        this_single_path = model_dir + f"/SingleHead_S{step}_{args.dataset_name}_Steps{args.num_steps}_{args.model_head}_Mlp{args.num_mlp_layers}.pth"
        args.learned_single_head_paths_list.append(this_single_path)

    # path to save single head
    args.save_single_path = model_dir + f"/SingleHead_S{args.current_step}_{args.dataset_name}_Steps{args.num_steps}_{args.model_head}_Mlp{args.num_mlp_layers}.pth"
    args.save_learned_single_path = weights_dir + f"/SingleHead_S{args.current_step}_{args.dataset_name}_Steps{args.num_steps}_{args.model_head}_Mlp{args.num_mlp_layers}.pth"

    args.save_joint_path = model_dir + f"/JointHead_S{args.current_step}_{args.dataset_name}_Steps{args.num_steps}_{args.model_head}_Mlp{args.num_mlp_layers}.pth"

    args.log_dir = model_dir + f'/{args.dataset_name}_{args.ucl_method}_S{str(args.current_step)}-{args.num_steps}_log.txt'
    sys.stdout = Logger(args.log_dir)

    print('log_dir=', args.log_dir)

    # WandB setting
    if args.mode == 'train':
        wandb_run_name = f'CaSSLe-Linear-iNCD-{args.ucl_method}_{args.dataset_name}_S{str(args.current_step)}-{args.num_steps}'
        wandb.init(project='CASSLE_iNCD',
                   entity=args.wandb_entity,
                   tags=[f'TotalStep={args.num_steps}', args.dataset_name, f'Steps={str(args.current_step)}',
                         args.model_name, args.ucl_method],
                   name=wandb_run_name,
                   mode=args.wandb_mode)

    # ----------------------
    # Experimental Setting Initialization
    # ----------------------
    # Dataset Split Params
    args.num_novel_interval = math.ceil(args.num_classes / args.num_steps)
    args.current_novel_start = args.num_novel_interval * args.current_step
    args.current_novel_end = args.num_novel_interval * (args.current_step + 1) \
        if args.num_novel_interval * (args.current_step + 1) <= args.num_classes \
        else args.num_classes
    args.num_novel_per_step = args.current_novel_end - args.current_novel_start

    # ViT DINO B/16 Params
    # Parameters
    # Parameters
    if 'vit' in args.model_name:
        args.image_size = 224
    elif 'resnet' in args.model_name:
        args.image_size = 64
    else:
        raise NotImplementedError

    # args.image_size = 224
    args.interpolation = 3
    args.crop_pct = 0.875
    args.pretrain_path = args.dino_pretrain_path
    args.feat_dim = 768
    args.mlp_out_dim = args.num_novel_per_step

    # ----------------------
    # Dataloaders Creation for this iNCD step
    # ----------------------
    data_factory = build_data(args)

    # Train loader
    ulb_train_loader = data_factory.get_dataloader(split='train', aug='twice', shuffle=True,
                                                   target_list=range(args.current_novel_start, args.current_novel_end))

    # Mixed-val loader
    if args.current_step > 0:
        ulb_all_prev_val_loader = data_factory.get_dataloader(split='train', aug=None, shuffle=False,
                                                              target_list=range(args.current_novel_start))
    else:
        ulb_all_prev_val_loader = None

    ulb_all_val_loader = data_factory.get_dataloader(split='train', aug=None, shuffle=False,
                                                     target_list=range(args.current_novel_end))

    val_split = args.val_split
    test_split = args.test_split

    # Mixed-test loader
    if args.current_step > 0:
        ulb_all_prev_test_loader = data_factory.get_dataloader(split='test', aug=None, shuffle=False,
                                                               target_list=range(args.current_novel_start))
    else:
        ulb_all_prev_test_loader = None

    ulb_all_test_loader = data_factory.get_dataloader(split='test', aug=None, shuffle=False,
                                                      target_list=range(args.current_novel_end))

    # Step-wise val/test loader list
    ulb_step_val_loader_list = []
    ulb_step_test_loader_list = []
    for s in range(1 + args.current_step):
        if (1 + s) < args.num_steps:
            this_ulb_val_loader = data_factory.get_dataloader(split=val_split, aug=None, shuffle=False,
                                                              target_list=range(s * args.num_novel_interval,
                                                                                (1 + s) * args.num_novel_interval))
            this_ulb_test_loader = data_factory.get_dataloader(split=test_split, aug=None, shuffle=False,
                                                               target_list=range(s * args.num_novel_interval,
                                                                                 (1 + s) * args.num_novel_interval))
        else:
            this_ulb_val_loader = data_factory.get_dataloader(split=val_split, aug=None, shuffle=False,
                                                              target_list=range(args.current_novel_start,
                                                                                args.current_novel_end))
            this_ulb_test_loader = data_factory.get_dataloader(split=test_split, aug=None, shuffle=False,
                                                               target_list=range(args.current_novel_start,
                                                                                 args.current_novel_end))

        ulb_step_val_loader_list.append(this_ulb_val_loader)
        ulb_step_test_loader_list.append(this_ulb_test_loader)

    # ----------------------
    # Direct Concat model creation:
    #   model: large-scale pre-trained backbone
    #   single_head: to-be-learned single head
    #   single_heads_list: previous single heads already learned
    #   joint_head: concat-dim head container
    # ----------------------
    model, single_head, single_heads_list, joint_head = build_direct_concat_model(args)

    print(args)

    # print("------> Backbone model:")
    # print(model)

    print("------> Single head:")
    print(single_head)

    print("------> Learned Single heads")
    for s_single in single_heads_list:
        print(s_single)

    print("------> Joint head:")
    print(joint_head)

    if args.mode == 'train':
        # Create Feature Replayer model
        sinkhorn = SinkhornKnopp(args)

        method = DirectConcat(model=model, single_head=single_head, learned_single_heads=single_heads_list,
                              joint_head=joint_head, sinkhorn=sinkhorn,
                              train_loader=ulb_train_loader,
                              ulb_step_val_list=ulb_step_val_loader_list,
                              ulb_all_prev_val=ulb_all_prev_val_loader,
                              ulb_all_val=ulb_all_val_loader,
                              ulb_step_test_list=ulb_step_test_loader_list,
                              ulb_all_prev_test=ulb_all_prev_test_loader,
                              ulb_all_test=ulb_all_test_loader)

        # Training
        method.train_single(args)
        # method.train_single_new(args)

        # Save trained student head weights
        #   |- save to experimental root
        method.save_single(path=args.save_single_path)
        method.save_joint_head(args, path=args.save_joint_path)
        #   |- save to weights warehouse
        method.save_single(path=args.save_learned_single_path)

        # Final test with test loader
        method.test(args)
    elif args.mode == 'eval':
        raise NotImplementedError
    else:
        raise NotImplementedError
