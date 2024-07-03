import torch
from utils.util import seed_torch
from utils.logging import Logger
import os
import sys
import wandb
import math

from models.build_weight_discrepancy import build_weight_discrepancy
from data.build_dataset import build_data
from methods.weight_discrepancy_method import WeightDiscrepancy
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
    parser.add_argument('--weight_decay', type=float, default=5e-5)

    # Strategy tricks
    parser.add_argument('--w_wd', type=str, default='0.01', help='weight for loss_weight_discrepancy')
    parser.add_argument('--epoch_wd', default=100, type=int)
    parser.add_argument('--conf_guided', action='store_true', default=False)
    parser.add_argument('--rampup_length', default=175, type=int)
    parser.add_argument('--rampup_coefficient', type=float, default=25)
    # UNO knobs
    parser.add_argument("--softmax_temp", default=0.1, type=float, help="softmax temperature")
    parser.add_argument("--threshold", default=0.5, type=float, help="threshold for hard pseudo-labeling")
    parser.add_argument("--num_iters_sk", default=3, type=int, help="number of iters for Sinkhorn")
    parser.add_argument("--epsilon_sk", default=0.05, type=float, help="epsilon for the Sinkhorn")
    parser.add_argument('--alpha', default=0.75, type=float)

    # Dataset Setting
    parser.add_argument('--dataset_root', type=str, default='./data/datasets/CIFAR/')
    parser.add_argument('--dataset_name', type=str, default='cifar100', help='options: cifar10, cifar100, tinyimagenet, cub200, herb19')
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
    parser.add_argument('--exp_root', type=str, default='./outputs/')
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
    model_dir = os.path.join(args.exp_root, f"{runner_name}_{args.model_name}_{args.dataset_name}_Steps{args.num_steps}_Epoch_wd_{args.epoch_wd}_Wwd_{args.w_wd}_ConfGuided_{args.conf_guided}")
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    # path to pre-trained teacher heads weights .pth file
    args.learned_single_head_paths_list = []
    for step in range(args.current_step):
        this_single_path = model_dir + f"/SingleHead_S{step}_{args.dataset_name}_Steps{args.num_steps}_{args.model_head}_Mlp{args.num_mlp_layers}.pth"
        args.learned_single_head_paths_list.append(this_single_path)

    args.save_single_path = model_dir + f"/SingleHead_S{args.current_step}_{args.dataset_name}_Steps{args.num_steps}_{args.model_head}_Mlp{args.num_mlp_layers}.pth"
    args.save_joint_path = model_dir + f"/JointHead_S{args.current_step}_{args.dataset_name}_Steps{args.num_steps}_{args.model_head}_Mlp{args.num_mlp_layers}.pth"

    args.log_dir = model_dir + f'/{args.dataset_name}_S{str(args.current_step)}-{args.num_steps}_Epoch_wd_{args.epoch_wd}_Wwd_{args.w_wd}_ConfGuided_{args.conf_guided}_log.txt'
    sys.stdout = Logger(args.log_dir)

    print('log_dir=', args.log_dir)

    # WandB setting
    if args.mode == 'train':
        wandb_run_name = f'Weight-Discrepancy_{args.model_name}_{args.dataset_name}_S{str(args.current_step)}-{args.num_steps}_Epoch_wd_{args.epoch_wd}_Wwd_{args.w_wd}_ConfGuided_{args.conf_guided}'
        wandb_tags = [f'TotalStep={args.num_steps}', args.dataset_name, f'Steps={str(args.current_step)}',
                      args.model_name, f'ConfGuided={args.conf_guided}', f'W_wd={args.w_wd}',
                      f'Epoch_wd={args.epoch_wd}']
        wandb.init(project='Our_WeightDiscrepancy',
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

    # No-augmentation Train loader for uncertainty calculation
    ulb_uncert_train_loader = data_factory.get_dataloader(split='train', aug=None, shuffle=False,
                                                          target_list=range(args.current_novel_start,
                                                                            args.current_novel_end))

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
    model, single_head, single_heads_list, joint_head = build_weight_discrepancy(args)

    print(args)

    print("------> Backbone model:")
    print(model)

    print("------> Teacher heads")
    for s_single in single_heads_list:
        print(s_single)

    print("------> Student head:")
    print(single_head)

    print("------> Joint head:")
    print(joint_head)

    if args.mode == 'train':
        # Create Feature Replayer model
        sinkhorn = SinkhornKnopp(args)

        # Weight Discprepancy learning strategy
        method = WeightDiscrepancy(model=model, single_head=single_head, learned_single_heads=single_heads_list,
                                   joint_head=joint_head,
                                   sinkhorn=sinkhorn,
                                   train_loader=ulb_train_loader,
                                   uncert_loader=ulb_uncert_train_loader,
                                   ulb_step_val_list=ulb_step_val_loader_list,
                                   ulb_all_prev_val=ulb_all_prev_val_loader,
                                   ulb_all_val=ulb_all_val_loader,
                                   ulb_step_test_list=ulb_step_test_loader_list,
                                   ulb_all_prev_test=ulb_all_prev_test_loader,
                                   ulb_all_test=ulb_all_test_loader)

        if args.current_step == 0:
            method.train_init(args)
        elif args.current_step > 0:
            method.train_il(args)

        # Save trained student head weights
        method.save_single(path=args.save_single_path)
        method.save_joint_head(args, path=args.save_joint_path)

        # Final test with test loader
        method.test(args)
    elif args.mode == 'eval':
        raise NotImplementedError
    else:
        raise NotImplementedError
