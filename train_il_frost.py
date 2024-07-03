import torch
import torch.nn.functional as F
from torch.optim import SGD, lr_scheduler
from sklearn.metrics.cluster import normalized_mutual_info_score as nmi_score
from sklearn.metrics import adjusted_rand_score as ari_score
from utils.util import cluster_acc, AverageMeter, seed_torch
from utils.logging import Logger
from tqdm import tqdm
import numpy as np
import os
import sys
import copy
import wandb
import math

from models.build_frost import build_frost_model
from data.build_dataset import build_data
from data.config_dataset import set_dataset_config
from utils.frost_feat_replay import FeatureReplayer

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='cluster', formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # Hyper-parameters Setting
    parser.add_argument('--labeling_method', type=str, default='rankstats', choices=['rankstats', 'sinkhorn'])
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--batch_size', default=256, type=int)

    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--gamma', type=float, default=0.1)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight_decay', type=float, default=1e-4)

    # UNO knobs
    parser.add_argument("--softmax_temp", default=0.1, type=float, help="softmax temperature")
    parser.add_argument("--num_iters_sk", default=3, type=int, help="number of iters for Sinkhorn")
    parser.add_argument("--epsilon_sk", default=0.05, type=float, help="epsilon for the Sinkhorn")
    parser.add_argument('--alpha', default=0.75, type=float)

    # Strategy tricks
    parser.add_argument('--step_size', default=70, type=int)
    parser.add_argument('--w_kd', type=float, default=10.0)
    parser.add_argument('--rampup_length', default=75, type=int)
    parser.add_argument('--rampup_coefficient', type=float, default=25)
    parser.add_argument('--increment_coefficient', type=float, default=0.05)
    parser.add_argument('--topk', default=5, type=int)
    parser.add_argument('--w_replay', type=float, default=1.0, help='weight for feature replay loss')

    # Dataset Setting
    parser.add_argument('--dataset_name', type=str, default='cifar100', choices=['cifar10', 'cifar100', 'tinyimagenet',
                                                                                 'cub200', 'herb19', 'scars',
                                                                                 'aircraft'])
    # parser.add_argument('--dataset_root', type=str, default='./data/datasets/CIFAR/')
    # parser.add_argument('--num_classes', default=100, type=int)
    parser.add_argument('--aug_type', type=str, default='vit_uno', choices=['vit_frost', 'vit_uno', 'resnet'])
    parser.add_argument('--num_workers', default=2, type=int)

    # Strategy Setting
    parser.add_argument('--num_steps', default=5, type=int)
    parser.add_argument('--current_step', default=0, type=int)

    # Model Config
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'eval'])
    parser.add_argument('--model_name', type=str, default='vit_dino')
    parser.add_argument('--grad_from_block', type=int, default=11)
    parser.add_argument('--num_mlp_layers', type=int, default=1)
    parser.add_argument('--dino_pretrain_path', type=str,
                        default='./models/dino_weights/dino_vitbase16_pretrain.pth')
    parser.add_argument('--model_head', type=str, default='LinearHead', choices=['LinearHead', 'DINOHead'])

    # Experimental Setting
    parser.add_argument('--seed', default=10, type=int)
    parser.add_argument('--exp_root', type=str, default='./outputs_frost/')

    parser.add_argument('--exp_marker', type=str, default='warmedup')
    parser.add_argument('--wandb_mode', type=str, default='online', choices=['online', 'offline', 'disabled'])
    parser.add_argument('--wandb_entity', type=str, default='oatmealliu')

    # ----------------------
    # Initial Configurations
    # ----------------------
    args = parser.parse_args()

    # init. dataset config.
    args = set_dataset_config(args)

    # init. config.
    args.cuda = torch.cuda.is_available()
    device = torch.device("cuda" if args.cuda else "cpu")
    args.device = torch.device("cuda" if args.cuda else "cpu")
    seed_torch(args.seed)

    # init. experimental output path
    runner_name = os.path.basename(__file__).split(".")[0]

    # set a dir name which can describe the experiment
    model_dir = os.path.join(args.exp_root, f"{runner_name}_Pseudo={args.labeling_method}_{args.dataset_name}_Steps{args.num_steps}_Block{args.grad_from_block}")
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    if args.labeling_method == 'rankstats':
        from methods.frost_method import FRoST as Method
        args.project_name = 'SOTA_FRoST'
    else:
        from methods.frost_sinkhorn import FRoST as Method
        args.project_name = 'SOTA_FRoST_Sinkhorn'

    if args.current_step == 0:
        # Single model and head saving path
        args.save_single_model_path = model_dir + f"/single_Backbone_S{args.current_step}_{args.dataset_name}_Steps{args.num_steps}_Block{args.grad_from_block}.pth"
        args.save_single_head_path = model_dir + f"/single_Head_S{args.current_step}_{args.dataset_name}_Steps{args.num_steps}_Block{args.grad_from_block}.pth"
    else:
        # Single model and head saving path
        args.save_single_model_path = model_dir + f"/single_Backbone_S{args.current_step}_{args.dataset_name}_Steps{args.num_steps}_Block{args.grad_from_block}.pth"
        args.save_single_head_path = model_dir + f"/single_Head_S{args.current_step}_{args.dataset_name}_Steps{args.num_steps}_Block{args.grad_from_block}.pth"

        # Joint model and head saving path
        args.save_joint_model_path = model_dir + f"/joint_Backbone_S{args.current_step}_{args.dataset_name}_Steps{args.num_steps}_Block{args.grad_from_block}.pth"
        args.save_joint_head_path = model_dir + f"/joint_Head_S{args.current_step}_{args.dataset_name}_Steps{args.num_steps}_Block{args.grad_from_block}.pth"

        # Previous learned single model and head path
        args.prev_single_backbone_paths_list = []
        args.prev_single_head_paths_list = []
        for step in range(args.current_step):
            this_single_backbone_path = model_dir + f"/single_Backbone_S{step}_{args.dataset_name}_Steps{args.num_steps}_Block{args.grad_from_block}.pth"
            this_single_head_path = model_dir + f"/single_Head_S{step}_{args.dataset_name}_Steps{args.num_steps}_Block{args.grad_from_block}.pth"
            args.prev_single_backbone_paths_list.append(this_single_backbone_path)
            args.prev_single_head_paths_list.append(this_single_head_path)

    args.log_dir = model_dir + f'/{args.dataset_name}_S{str(args.current_step)}-{args.num_steps}_{args.model_head}_Mlp{args.num_mlp_layers}_Block{args.grad_from_block}_log.txt'
    sys.stdout = Logger(args.log_dir)

    print('log_dir=', args.log_dir)

    # WandB setting
    wandb_tags = [args.dataset_name, f'TotalStep={args.num_steps}', f'Steps={str(args.current_step)}', "FRoST",
                  f'Block={args.grad_from_block}', f'device={args.device}', f'Pseudo={args.labeling_method}']
    wandb_run_name = f'FRoST_{args.dataset_name}_S{str(args.current_step)}/{args.num_steps}_Block{args.grad_from_block}'
    wandb.init(project=args.project_name,
               entity=args.wandb_entity,
               tags=wandb_tags,
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
    args.image_size = 224
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
        ulb_all_prev_test_loader = data_factory.get_dataloader(split=test_split, aug=None, shuffle=False,
                                                               target_list=range(args.current_novel_start))
    else:
        ulb_all_prev_test_loader = None

    ulb_all_test_loader = data_factory.get_dataloader(split=test_split, aug=None, shuffle=False,
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

    if args.mode == 'train' and args.current_step == 0:
        print("Init-Training")
        model, single_head = build_frost_model(args)

        # FRoST method object
        method = Method(model=model, single_head=single_head,
                        joint_head=None, prev_pair_list=None, feat_replayer=None,
                        train_loader=ulb_train_loader,
                        ulb_step_val_list=ulb_step_val_loader_list,
                        ulb_all_prev_val=ulb_all_prev_val_loader,
                        ulb_all_val=ulb_all_val_loader,
                        ulb_step_test_list=ulb_step_test_loader_list,
                        ulb_all_prev_test=ulb_all_prev_test_loader,
                        ulb_all_test=ulb_all_test_loader)

        # Training
        #   |- Train
        method.train_init(args)
        #   |- Save learned model and head
        method.save_single(model_path=args.save_single_model_path, head_path=args.save_single_head_path)
        #   |- Test evaluation
        method.test_init(args)
    elif args.mode == 'train' and args.current_step > 0:
        print("Incremental-Training")

        model, single_head, joint_head, prev_pair_list = build_frost_model(args)

        # Create Feature Replayer model
        feat_replayer = FeatureReplayer(args, prev_pair_list, data_factory)

        # FRoST method object
        method = Method(model=model, single_head=single_head,
                        joint_head=joint_head, prev_pair_list=prev_pair_list, feat_replayer=feat_replayer,
                        train_loader=ulb_train_loader,
                        ulb_step_val_list=ulb_step_val_loader_list,
                        ulb_all_prev_val=ulb_all_prev_val_loader,
                        ulb_all_val=ulb_all_val_loader,
                        ulb_step_test_list=ulb_step_test_loader_list,
                        ulb_all_prev_test=ulb_all_prev_test_loader,
                        ulb_all_test=ulb_all_test_loader)

        # Training
        #   |- Train
        method.train_IL(args)
        #   |- Save learned models and heads
        method.save_single(model_path=args.save_single_model_path, head_path=args.save_single_head_path)
        method.save_joint(model_path=args.save_joint_head_path, head_path=args.save_joint_model_path)
        #   |- Test evaluation
        method.test_IL(args)
    elif args.mode == 'eval':
        raise NotImplementedError
    else:
        raise NotImplementedError
