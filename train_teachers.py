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

from models.build_teacher_student_professor import build_teacher
from data.build_dataset import build_data
from utils.sinkhorn_knopp import SinkhornKnopp
from data.config_dataset import set_dataset_config

from methods.teacher_learning import train_Teacher
from methods.testers import test_single

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='cluster', formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # Hyper-parameters Setting
    parser.add_argument('--epochs', default=200, type=int)
    parser.add_argument('--batch_size', default=128, type=int)

    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--gamma', type=float, default=0.1)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight_decay', type=float, default=1e-4)

    parser.add_argument('--l2_single_cls', action='store_true', default=False,
                        help='L2 normalize single classifier weights before forward-prop')
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
    parser.add_argument('--aug_type', type=str, default='vit_uno', choices=['vit_frost', 'vit_uno', 'resnet'])
    parser.add_argument('--num_workers', default=8, type=int)

    # Strategy Setting
    parser.add_argument('--num_steps', default=10, type=int)
    parser.add_argument('--current_step', default=0, type=int)

    # Model Config
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'eval'])
    parser.add_argument('--model_name', type=str, default='vit_dino')
    parser.add_argument('--grad_from_block', type=int, default=12)  # 12->do not fine tune backbone at all
    parser.add_argument('--num_mlp_layers', type=int, default=3)  # 12->do not fine tune backbone at all
    parser.add_argument('--dino_pretrain_path', type=str,
                        default='./models/dino_weights/dino_vitbase16_pretrain.pth')
    parser.add_argument('--model_head', type=str, default='LinearHead', choices=['LinearHead', 'DINOHead'])

    # Experimental Setting
    parser.add_argument('--seed', default=1, type=int)
    parser.add_argument('--exp_root', type=str, default='./models/teacher_weights/')
    parser.add_argument('--exp_marker', type=str, default='vit_dino_expt')
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
    model_dir = os.path.join(args.exp_root, f"{args.dataset_name}_Steps{args.num_steps}_{args.model_head}_Mlp{args.num_mlp_layers}")
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    # Single teacher dir
    args.teacher_head_dir = model_dir + f"/teacherHead_S{args.current_step}_{args.dataset_name}_Steps{args.num_steps}_{args.model_head}_Mlp{args.num_mlp_layers}.pth"

    args.log_dir = model_dir + f'/{args.dataset_name}_S{str(args.current_step)}-{args.num_steps}_{args.model_head}_Mlp{args.num_mlp_layers}_log.txt'
    sys.stdout = Logger(args.log_dir)

    print('log_dir=', args.log_dir)

    # WandB setting
    if args.mode == 'train':
        wandb_run_name = f'{args.dataset_name}_S{str(args.current_step)}-{args.num_steps}_{args.model_head}_Mlp{args.num_mlp_layers}'
        wandb.init(project='MSc-iNCD-University',
                   entity=args.wandb_entity,
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

    unlabeled_train_loader = data_factory.get_dataloader(split='train', aug='twice', shuffle=True,
                                                         target_list=range(args.current_novel_start,
                                                                           args.current_novel_end))

    ulb_val_loader = data_factory.get_dataloader(split='train', aug=None, shuffle=False,
                                                 target_list=range(args.current_novel_start, args.current_novel_end))

    ulb_test_loader = data_factory.get_dataloader(split='test', aug=None, shuffle=False,
                                                  target_list=range(args.current_novel_start, args.current_novel_end))

    # ----------------------
    # ViT Model and DINO Projection Head Creations for this iNCD step
    # ----------------------
    model, teacher = build_teacher(args)

    print(args)
    print(model)
    print(teacher)

    if args.mode == 'train':
        # Sinkhorn
        sinkhorn = SinkhornKnopp(args)

        teacher = train_Teacher(model, teacher, sinkhorn, unlabeled_train_loader, ulb_val_loader, args)

        torch.save(teacher.state_dict(), args.teacher_head_dir)

        print("Teacher Head for this NCD step saved to {}.".format(args.teacher_head_dir))

        acc_ulb_test_w_clustering = test_single(model, teacher, ulb_test_loader, args, cluster=True)

        print('\n===========================================')
        print(f"Acc_Test_ulb for {args.dataset_name} [{args.current_novel_start}, {args.current_novel_end}] = {acc_ulb_test_w_clustering} (w/ clustering)")
        print('===========================================')
    elif args.mode == 'eval':
        raise NotImplementedError
    else:
        raise NotImplementedError
