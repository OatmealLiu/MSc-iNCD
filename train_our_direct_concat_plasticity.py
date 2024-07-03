import torch
from utils.util import seed_torch
from utils.logging import Logger
import os
import sys
import wandb
import math

from utils.sinkhorn_knopp import SinkhornKnopp
from models.build_plasticity import build_plasticity
from data.build_dataset import build_data
from methods.direct_concat_plasticity import DirectConcatPlasticity
from data.config_dataset import set_dataset_config

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='cluster', formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # Hyper-parameters Setting
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--unlock_epoch', default=100, type=int)
    parser.add_argument('--batch_size', default=128, type=int)

    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--gamma', type=float, default=0.1)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight_decay', type=float, default=5e-5)

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
    parser.add_argument('--num_workers', default=2, type=int)

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
    parser.add_argument('--lock_head', action='store_true', default=False)

    # Experimental Setting
    parser.add_argument('--seed', default=10, type=int)
    parser.add_argument('--exp_root', type=str, default='./outputs/')

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
    args.device_count = torch.cuda.device_count()
    seed_torch(args.seed)

    # init. experimental output path
    runner_name = os.path.basename(__file__).split(".")[0]

    # set a dir name which can describe the experiment
    model_dir = os.path.join(args.exp_root, f"{runner_name}_{args.dataset_name}_Steps{args.num_steps}_{args.model_name}_{args.model_head}_GradBlock{args.grad_from_block}_LockHead_{args.lock_head}")
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    # Path to load the incrementally fine-tune model
    if args.current_step == 0:
        args.load_model_path = args.dino_pretrain_path
    else:
        args.load_model_path = model_dir + f"/single_Backbone_S{args.current_step-1}_{args.dataset_name}_Steps{args.num_steps}_{args.model_name}_{args.model_head}_GradBlock{args.grad_from_block}_LockHead_{args.lock_head}.pth"

    # Paths to load individual model and corresponding head
    args.learned_model_paths_list = []
    args.learned_single_head_paths_list = []
    for step in range(args.current_step):
        this_backbone_path = model_dir + f"/single_Backbone_S{step}_{args.dataset_name}_Steps{args.num_steps}_{args.model_name}_{args.model_head}_GradBlock{args.grad_from_block}_LockHead_{args.lock_head}.pth"
        this_head_path = model_dir + f"/single_Head_S{step}_{args.dataset_name}_Steps{args.num_steps}_{args.model_name}_{args.model_head}_GradBlock{args.grad_from_block}_LockHead_{args.lock_head}.pth"
        args.learned_model_paths_list.append(this_backbone_path)
        args.learned_single_head_paths_list.append(this_head_path)

    # Paths to save the current fine-tune model backbone and single head
    args.save_model_path = model_dir + f"/single_Backbone_S{args.current_step}_{args.dataset_name}_Steps{args.num_steps}_{args.model_name}_{args.model_head}_GradBlock{args.grad_from_block}_LockHead_{args.lock_head}.pth"
    args.save_single_head_path = model_dir + f"/single_Head_S{args.current_step}_{args.dataset_name}_Steps{args.num_steps}_{args.model_name}_{args.model_head}_GradBlock{args.grad_from_block}_LockHead_{args.lock_head}.pth"
    args.save_joint_head_path = model_dir + f"/joint_Head_S{args.current_step}_{args.dataset_name}_Steps{args.num_steps}_{args.model_name}_{args.model_head}_GradBlock{args.grad_from_block}_LockHead_{args.lock_head}.pth"

    args.log_dir = model_dir + f'/Plasticity_{args.dataset_name}_S{str(args.current_step)}-{args.num_steps}_{args.model_name}_{args.model_head}_GradBlock{args.grad_from_block}_LockHead_{args.lock_head}_log.txt'
    sys.stdout = Logger(args.log_dir)

    print('log_dir=', args.log_dir)

    # WandB setting
    wandb_tags = [f'TotalStep={args.num_steps}', args.dataset_name, args.model_name, f'LockHead={args.lock_head}',
                  f'Steps={str(args.current_step)}', f'GradBlock={args.grad_from_block}',
                  f'UnlockEpoch={args.unlock_epoch}']
    wandb_run_name = f'Plasticity_{args.dataset_name}_S{str(args.current_step)}-{args.num_steps}_{args.model_name}_{args.model_head}_GradBlock{args.grad_from_block}_LockHead_{args.lock_head}'
    wandb.init(project='Our_plasticity',
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

    # # Train loader list for joint training
    # ulb_train_loader_list = []
    # for s in range(1 + args.current_step):
    #     if (1 + s) < args.num_steps:
    #         s_ulb_train_loader = data_factory.get_dataloader(split='train', aug='twice', shuffle=True,
    #                                                          target_list=range(s * args.num_novel_interval,
    #                                                                            (1 + s) * args.num_novel_interval))
    #     else:
    #         s_ulb_train_loader = data_factory.get_dataloader(split='train', aug='twice', shuffle=True,
    #                                                          target_list=range(args.current_novel_start,
    #                                                                            args.current_novel_end))
    #     ulb_train_loader_list.append(s_ulb_train_loader)

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

    if args.mode == 'train':
        model, single_head, learned_model_head_pair_list, joint_head = build_plasticity(args)

        # Create SinkhornKnopp pseudo-labeling algo.
        sinkhorn = SinkhornKnopp(args)

        # TeacherStudent learning strategy
        method = DirectConcatPlasticity(model=model, single_head=single_head, joint_head=joint_head,
                                        prev_pair_list=learned_model_head_pair_list,
                                        sinkhorn=sinkhorn,
                                        train_loader=ulb_train_loader,
                                        ulb_step_val_list=ulb_step_val_loader_list,
                                        ulb_all_prev_val=ulb_all_prev_val_loader,
                                        ulb_all_val=ulb_all_val_loader,
                                        ulb_step_test_list=ulb_step_test_loader_list,
                                        ulb_all_prev_test=ulb_all_prev_test_loader,
                                        ulb_all_test=ulb_all_test_loader)

        # Learning
        method.train(args)

        # Save
        method.save_backbone(args.save_model_path)
        method.save_single_head(args.save_single_head_path)
        method.save_joint_head(args.save_joint_head_path)

        # Testing
        method.test(args)
    elif args.mode == 'eval':
        raise NotImplementedError
    else:
        raise NotImplementedError
