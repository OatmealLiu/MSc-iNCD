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

from models.build_rankstats import build_eval_rankstats
from data.build_dataset import build_data
from methods.rankstats_learning import RankStats
from utils.sinkhorn_knopp import SinkhornKnopp

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='cluster', formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # Hyper-parameters Setting
    parser.add_argument('--epochs', default=200, type=int)
    parser.add_argument('--batch_size', default=256, type=int)

    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--gamma', type=float, default=0.1)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight_decay', type=float, default=1e-4)

    # Strategy tricks
    parser.add_argument('--use_norm', action='store_true', default=False,
                        help='L2 normalize single classifier weights before forward-prop')
    parser.add_argument("--softmax_temp", default=1.0, type=float, help="softmax temperature")
    parser.add_argument('--topk', default=5, type=int)
    parser.add_argument("--w_bce", default=1.0, type=float, help="weight for bce loss")
    parser.add_argument("--w_entropy", default=10.0, type=float, help="weight for entropy loss")
    parser.add_argument('--step_size', default=170, type=int)

    # Dataset Setting
    parser.add_argument('--dataset_root', type=str, default='./data/datasets/CIFAR/')
    parser.add_argument('--dataset_name', type=str, default='cifar100', help='options: cifar10, cifar100, tinyimagenet')
    parser.add_argument('--num_classes', default=100, type=int)
    parser.add_argument('--aug_type', type=str, default='vit_uno', choices=['vit_frost', 'vit_uno', 'resnet'])
    parser.add_argument('--num_workers', default=8, type=int)

    # Strategy Setting
    parser.add_argument('--num_steps', default=10, type=int)
    parser.add_argument('--current_step', default=0, type=int)

    # Model Config
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'eval'])
    parser.add_argument('--model_name', type=str, default='vit_dino')
    parser.add_argument('--grad_from_block', type=int, default=12)  # 12->do not fine tune backbone at all
    parser.add_argument('--num_mlp_layers', type=int, default=1)  # 12->do not fine tune backbone at all
    parser.add_argument('--dino_pretrain_path', type=str,
                        default='./models/dino_weights/dino_vitbase16_pretrain.pth')
    parser.add_argument('--model_head', type=str, default='LinearHead', choices=['LinearHead', 'DINOHead'])

    # Experimental Setting
    parser.add_argument('--seed', default=10, type=int)
    parser.add_argument('--exp_root', type=str, default='./models/teacher_weights/')
    parser.add_argument('--exp_marker', type=str, default='nonsense_expt')
    parser.add_argument('--wandb_mode', type=str, default='online', choices=['online', 'offline', 'disabled'])
    parser.add_argument('--wandb_entity', type=str, default='oatmealliu')

    # ----------------------
    # Initial Configurations
    # ----------------------
    args = parser.parse_args()

    # init. config.
    args.cuda = torch.cuda.is_available()
    device = torch.device("cuda" if args.cuda else "cpu")
    args.device = torch.device("cuda" if args.cuda else "cpu")
    seed_torch(args.seed)

    # init. experimental output path
    runner_name = os.path.basename(__file__).split(".")[0]

    # set a dir name which can describe the experiment
    model_dir = os.path.join(args.exp_root, f"{runner_name}_{args.dataset_name}_Steps{args.num_steps}_{args.model_head}_NormCls_{args.use_norm}_Aug_{args.aug_type}_topK_{args.topk}_wBCE_{args.w_bce}_wEntropy_{args.w_entropy}B")
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    # path to pre-trained teacher heads weights .pth file
    # studentHead_S0_cifar100_Steps4_LinearHead_Mlp1.pth
    args.pretrained_teacher_head_paths_list = []
    for step in range(args.current_step):
        this_teacher_path = model_dir + f"/studentHead_S{step}_{args.dataset_name}_Steps{args.num_steps}_{args.model_head}_NormCls_{args.use_norm}_Aug_{args.aug_type}.pth"
        args.pretrained_teacher_head_paths_list.append(this_teacher_path)

    # path to save student
    args.save_student_path = model_dir + f"/studentHead_S{args.current_step}_{args.dataset_name}_Steps{args.num_steps}_{args.model_head}_NormCls_{args.use_norm}_Aug_{args.aug_type}.pth"
    args.save_joint_path = model_dir + f"/jointHead_S{args.current_step}_{args.dataset_name}_Steps{args.num_steps}_{args.model_head}_NormCls_{args.use_norm}_Aug_{args.aug_type}.pth"

    args.log_dir = model_dir + f'/{args.dataset_name}_S{str(args.current_step)}-{args.num_steps}_{args.model_head}_NormCls_{args.use_norm}_Aug_{args.aug_type}_topK_{args.topk}_wBCE_{args.w_bce}_wEntropy_{args.w_entropy}_log.txt'
    sys.stdout = Logger(args.log_dir)

    print('log_dir=', args.log_dir)

    # WandB setting
    if args.mode == 'train':
        wandb_tags = [f'TotalStep={args.num_steps}', args.dataset_name, f'Steps={str(args.current_step)}',
                      f'WeightNorm={args.use_norm}', f'W_bce={args.w_bce}', f'W_entropy={args.w_entropy}']

        wandb_run_name = f'Direct-RankStats_{args.dataset_name}_S{str(args.current_step)}-{args.num_steps}_{args.model_head}_NormCls_{args.use_norm}_Aug_{args.aug_type}_topK_{args.topk}_wBCE_{args.w_bce}_wEntropy_{args.w_entropy}'
        wandb.init(project='Ours_RankStats',
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
            this_ulb_val_loader = data_factory.get_dataloader(split='train', aug=None, shuffle=False,
                                                              target_list=range(s * args.num_novel_interval,
                                                                                (1 + s) * args.num_novel_interval))
            this_ulb_test_loader = data_factory.get_dataloader(split='test', aug=None, shuffle=False,
                                                               target_list=range(s * args.num_novel_interval,
                                                                                 (1 + s) * args.num_novel_interval))
        else:
            this_ulb_val_loader = data_factory.get_dataloader(split='train', aug=None, shuffle=False,
                                                              target_list=range(args.current_novel_start,
                                                                                args.current_novel_end))
            this_ulb_test_loader = data_factory.get_dataloader(split='test', aug=None, shuffle=False,
                                                               target_list=range(args.current_novel_start,
                                                                                 args.current_novel_end))

        ulb_step_val_loader_list.append(this_ulb_val_loader)
        ulb_step_test_loader_list.append(this_ulb_test_loader)

    # ----------------------
    # Teacher Student model creation:
    #   model: large-scale pre-trained backbone
    #   teachers_list: pre-trained single head model
    #   student: joint head model
    # ----------------------
    model, teachers_list, student, joint_head = build_eval_rankstats(args)

    print(args)

    print("------> Backbone model:")
    print(model)

    print("------> Teacher heads")
    for teacher in teachers_list:
        print(teacher)

    print("------> Student head:")
    print(student)

    print("------> Joint head:")
    print(joint_head)

    if args.mode == 'train':
        # TeacherStudent learning strategy
        method = RankStats(model=model, teachers_list=teachers_list, student=student, joint_head=joint_head,
                           train_loader=ulb_train_loader,
                           ulb_step_val_list=ulb_step_val_loader_list,
                           ulb_all_prev_val=ulb_all_prev_val_loader,
                           ulb_all_val=ulb_all_val_loader,
                           ulb_step_test_list=ulb_step_test_loader_list,
                           ulb_all_prev_test=ulb_all_prev_test_loader,
                           ulb_all_test=ulb_all_test_loader)

        # Final test with test loader
        method.eval(args)
    elif args.mode == 'eval':
        raise NotImplementedError
    else:
        raise NotImplementedError
