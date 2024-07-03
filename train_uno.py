import torch

from utils.util import seed_torch
from utils.logging import Logger
import os
import sys
import copy
import wandb
import math

from utils.sinkhorn_knopp import SinkhornKnopp
from models.build_uno import build_uno
from data.build_dataset import build_data
from data.config_dataset import set_dataset_config

from methods.uno import UNO

import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='cluster', formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # Hyper-parameters Setting
    parser.add_argument('--epochs_pretrain', default=200, type=int)
    parser.add_argument('--epochs_ncd', default=200, type=int)
    parser.add_argument('--batch_size', default=256, type=int)

    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--gamma', type=float, default=0.1)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight_decay', type=float, default=5e-5)

    # UNO knobs
    parser.add_argument("--softmax_temp", default=0.1, type=float, help="softmax temperature")
    parser.add_argument("--num_iters_sk", default=3, type=int, help="number of iters for Sinkhorn")
    parser.add_argument("--epsilon_sk", default=0.05, type=float, help="epsilon for the Sinkhorn")
    parser.add_argument('--alpha', default=0.75, type=float)

    # Dataset Setting
    parser.add_argument('--dataset_name', type=str, default='cifar100', choices=['cifar10', 'cifar100', 'tinyimagenet',
                                                                                 'cub200', 'herb19', 'scars',
                                                                                 'aircraft'])
    parser.add_argument('--num_steps', default=2, type=int)

    # parser.add_argument('--dataset_root', type=str, default='./data/datasets/CIFAR/')
    # parser.add_argument('--num_classes', default=100, type=int)
    # parser.add_argument('--num_base', default=80, type=int)
    # parser.add_argument('--num_novel', default=20, type=int)

    parser.add_argument('--aug_type', type=str, default='vit_uno', choices=['vit_frost', 'vit_uno', 'resnet',
                                                                            'vit_uno_clip'])
    parser.add_argument('--base_aug', type=str, default='once', choices=['plain', 'once', 'supervised'])

    parser.add_argument('--num_workers', default=8, type=int)

    # Model Config
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'eval'])
    parser.add_argument('--model_name', type=str, default='vit_dino', choices=['resnet18_plain', 'resnet18_imagenet1k',
                                                                               'vit_plain', 'vit_dino',
                                                                               'resnet50_plain', 'resnet50_dino'])
    parser.add_argument('--grad_from_block', type=int, default=11)  # 12->do not fine tune backbone at all
    parser.add_argument('--num_mlp_layers', type=int, default=1)  # 12->do not fine tune backbone at all
    parser.add_argument('--dino_pretrain_path', type=str,
                        default='./models/dino_weights/dino_vitbase16_pretrain.pth')
    parser.add_argument('--model_head', type=str, default='LinearHead', choices=['LinearHead', 'DINOHead'])
    parser.add_argument('--lock_ncd_stage', type=str, default='unlock', choices=['unlock', 'lock'])

    # Experimental Setting
    parser.add_argument('--seed', default=10, type=int)
    parser.add_argument('--exp_root', type=str, default='./outputs_uno_study/')
    parser.add_argument('--wandb_mode', type=str, default='online', choices=['online', 'offline', 'disabled'])
    parser.add_argument('--wandb_entity', type=str, default='oatmealliu')

    # ----------------------
    # Initial Configurations
    # ----------------------
    args = parser.parse_args()

    # init. dataset config.
    args = set_dataset_config(args, setting='ncd')

    # init. config.
    args.cuda = torch.cuda.is_available()
    device = torch.device("cuda" if args.cuda else "cpu")
    args.device = torch.device("cuda" if args.cuda else "cpu")
    args.device_count = torch.cuda.device_count()
    seed_torch(args.seed)

    # init. experimental output path
    runner_name = os.path.basename(__file__).split(".")[0]

    # Experimental Dir.
    model_dir = os.path.join(args.exp_root, f"{runner_name}_{args.dataset_name}-{args.num_classes}({args.num_base}-{args.num_novel})_{args.model_name}_base_aug={args.base_aug}")
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    # path to save single head
    args.save_backbone_path = model_dir + f"/backbone_{args.dataset_name}-{args.num_base}-{args.num_novel}_{args.model_name}.pth"
    args.save_head_base_path = model_dir + f"/headBase_{args.dataset_name}-{args.num_base}-{args.num_novel}_{args.model_name}.pth"
    args.save_head_novel_path = model_dir + f"/headNovel_{args.dataset_name}-{args.num_base}-{args.num_novel}_{args.model_name}.pth"
    args.save_head_joint_path = model_dir + f"/headJoint_{args.dataset_name}-{args.num_base}-{args.num_novel}_{args.model_name}.pth"

    args.log_dir = model_dir + f'/{args.dataset_name}-{args.num_base}-{args.num_novel}_{args.model_name}_base_aug={args.base_aug}_log.txt'
    sys.stdout = Logger(args.log_dir)

    print('log_dir=', args.log_dir)

    # WandB setting
    if args.mode == 'train':
        wandb_run_name = f'UNO-Study_{args.dataset_name}-base_aug={args.base_aug}-lock_ncd_stage-{args.lock_ncd_stage}-{args.num_classes}({args.num_base}-{args.num_novel})_{args.model_name}_base_aug={args.base_aug}'
        wandb.init(project='UNO_study',
                   entity=args.wandb_entity,
                   tags=[args.model_name, args.dataset_name, f'num_classes={args.num_classes}',
                         f'num_base={args.num_base}', f'num_novel={args.num_novel}', f'base_aug={args.base_aug}',
                         f'lock_ncd_stage={args.lock_ncd_stage}'],
                   name=wandb_run_name,
                   mode=args.wandb_mode)

    # ----------------------
    # Experimental Setting Initialization
    # ----------------------
    # Parameters
    if args.model_name in ['vit_plain', 'vit_dino']:
        args.image_size = 224
    else:
        args.image_size = 64

    args.interpolation = 3
    args.crop_pct = 0.875
    args.pretrain_path = args.dino_pretrain_path
    args.feat_dim = 768
    args.mlp_out_dim = args.num_classes

    # ----------------------
    # Dataloaders Creation for this iNCD step
    # ----------------------
    data_factory = build_data(args)

    # labeled train loader
    lb_train_loader = data_factory.get_dataloader(split='train', aug=args.base_aug, shuffle=True,
                                                  target_list=range(args.num_base))
    # unlabeled train loader
    ulb_train_loader = data_factory.get_dataloader(split='train', aug='twice', shuffle=True,
                                                   target_list=range(args.num_base, args.num_classes))

    val_split = args.val_split
    test_split = args.test_split

    # labeled val loader
    val_loader_base = data_factory.get_dataloader(split=val_split, aug=None, shuffle=False,
                                                  target_list=range(args.num_base))
    # unlabeled val loader
    val_loader_novel = data_factory.get_dataloader(split=val_split, aug=None, shuffle=False,
                                                   target_list=range(args.num_base, args.num_classes))

    # labeled test loaders
    test_loader_base = data_factory.get_dataloader(split=test_split, aug=None, shuffle=False,
                                                   target_list=range(args.num_base))
    # unlabeled test loaders
    test_loader_novel = data_factory.get_dataloader(split=test_split, aug=None, shuffle=False,
                                                    target_list=range(args.num_base, args.num_classes))
    # # all test loaders
    # test_loader_all = data_factory.get_dataloader(split=test_split, aug=None, shuffle=False,
    #                                               target_list=range(args.num_classes))

    # ----------------------
    # Create backbone and classifier
    # ----------------------
    model, head_base, head_novel, head_joint = build_uno(args)

    print(args)

    print("------> Backbone model:")
    print(model)

    print("------> Head_base:")
    print(head_base)

    print("------> Head_novel:")
    print(head_novel)

    print("------> Head_joint:")
    print(head_joint)

    if args.mode == 'train':
        # Create Feature Replayer model
        sinkhorn = SinkhornKnopp(args)

        method = UNO(model=model, head_base=head_base, head_novel=head_novel, head_joint=head_joint,
                     sinkhorn=sinkhorn,
                     lb_train_loader=lb_train_loader, ulb_train_loader=ulb_train_loader,
                     val_loader_base=val_loader_base, val_loader_novel=val_loader_novel,
                     test_loader_base=test_loader_base, test_loader_novel=test_loader_novel)

        # Training
        #   |- supervised pre-training
        method.train_pretrain(args)
        method.test_pretrain(args)
        method.save_backbone(path=args.save_backbone_path)
        method.save_head_base(path=args.save_head_base_path)

        #   |- novel class discovery (joint)
        method.train_ncd(args)
        method.test_ncd(args)
        method.save_backbone(path=args.save_backbone_path)
        method.save_head_base(path=args.save_head_base_path)
        method.save_head_novel(path=args.save_head_novel_path)
        method.save_head_joint(path=args.save_head_joint_path)

    elif args.mode == 'eval':
        raise NotImplementedError
    else:
        raise NotImplementedError
