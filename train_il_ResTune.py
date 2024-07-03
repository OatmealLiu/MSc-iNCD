import torch
from utils.util import seed_torch
from utils.logging import Logger
import os
import sys
import copy
import wandb
import math

from utils.sinkhorn_knopp import SinkhornKnopp
from utils.step_tool import StepResults

from data.build_dataset import build_data
from data.config_dataset import set_dataset_config
from models.build_ResTune import build_restune_model
from methods.ResTune import ResTune

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='cluster', formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # Hyper-parameters Setting
    parser.add_argument('--epochs_warmup', default=100, type=int)
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--batch_size', default=16, type=int)

    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--gamma', type=float, default=0.1)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight_decay', type=float, default=5e-5)

    # UNO knobs
    parser.add_argument("--softmax_temp", default=0.1, type=float, help="softmax temperature")
    parser.add_argument("--num_iters_sk", default=3, type=int, help="number of iters for Sinkhorn")
    parser.add_argument("--epsilon_sk", default=0.05, type=float, help="epsilon for the Sinkhorn")
    parser.add_argument('--alpha', default=0.75, type=float)

    # Knowledge Distillation weights
    parser.add_argument("--w_kd", default=1.0, type=float, help="weight for KD loss")

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
    parser.add_argument('--num_steps', default=5, type=int)

    # Model Config
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'eval'])
    parser.add_argument('--model_name', type=str, default='vit_dino', choices=['vit_dino', 'clip', 'resnet50_dino',
                                                                               'resnet18_imagenet1k'])
    parser.add_argument('--grad_from_block', type=int, default=10)
    parser.add_argument('--dino_pretrain_path', type=str, default='./models/dino_weights/dino_vitbase16_pretrain.pth')

    # Experimental Setting
    parser.add_argument('--seed', default=10, type=int)

    parser.add_argument('--exp_root', type=str, default='./outputs_ResTune/')

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

    # Experimental Dir.
    model_dir = os.path.join(args.exp_root, f"{runner_name}_{args.model_name}_{args.dataset_name}_Steps{args.num_steps}_Wkd_{args.w_kd}_GradBlock_{args.grad_from_block}")
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    args.log_dir = model_dir + f'/{runner_name}_{args.model_name}_{args.dataset_name}_Steps{args.num_steps}_Wkd_{args.w_kd}_GradBlock_{args.grad_from_block}_log.txt'
    sys.stdout = Logger(args.log_dir)

    print('log_dir=', args.log_dir)

    # WandB setting
    # if args.mode == 'train':
    #     wandb_run_name = f'ResTune_{args.dataset_name}_{args.model_name}_{args.num_steps}-Steps_Wkd_{args.w_kd}_GradBlock_{args.grad_from_block}'
    #     wandb.init(project='SOTA_ResTune',
    #                entity=args.wandb_entity,
    #                tags=[f'TotalStep={args.num_steps}', args.dataset_name, args.model_name,
    #                      f'Wkd={args.w_kd}', f'GradBlock={args.grad_from_block}', f'device={args.device}'],
    #                name=wandb_run_name,
    #                mode=args.wandb_mode)

    # ----------------------
    # Experimental Setting Initialization
    # ----------------------
    # Dataset Split Params
    args.num_novel_interval = math.ceil(args.num_classes / args.num_steps)
    # args.current_novel_start = args.num_novel_interval * args.current_step
    # args.current_novel_end = args.num_novel_interval * (args.current_step + 1) \
    #     if args.num_novel_interval * (args.current_step + 1) <= args.num_classes \
    #     else args.num_classes
    # args.num_novel_per_step = args.current_novel_end - args.current_novel_start

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

    # ----------------------
    # Dataloaders Creation for this iNCD step
    # ----------------------
    data_factory = build_data(args)

    val_split = args.val_split
    test_split = args.test_split

    # Train loader
    step_train_loader_list = []
    step_val_loader_list = []
    step_test_loader_list = []

    prev_val_loader_list = []
    all_val_loader_list = []

    prev_test_loader_list = []
    all_test_loader_list = []

    # Generate step-wise data loader
    for s in range(args.num_steps):
        start_class = s * args.num_novel_interval
        end_class = (1+s) * args.num_novel_interval

        # D_train_s
        step_train_loader = data_factory.get_dataloader(split='train', aug='twice', shuffle=True,
                                                        target_list=range(start_class, end_class))
        step_train_loader_list.append(step_train_loader)

        # D_val_s
        step_val_loader = data_factory.get_dataloader(split=val_split, aug=None, shuffle=False,
                                                      target_list=range(start_class, end_class))
        step_val_loader_list.append(step_val_loader)

        # D_test_s
        step_test_loader = data_factory.get_dataloader(split=test_split, aug=None, shuffle=False,
                                                       target_list=range(start_class, end_class))
        step_test_loader_list.append(step_test_loader)

        if s > 0:
            # D_prev_val_s
            prev_val_loader = data_factory.get_dataloader(split=val_split, aug=None, shuffle=False,
                                                          target_list=range(start_class))
            prev_val_loader_list.append(prev_val_loader)

            # D_prev_test_s
            prev_test_loader = data_factory.get_dataloader(split=test_split, aug=None, shuffle=False,
                                                           target_list=range(start_class))
            prev_test_loader_list.append(prev_test_loader)

        # D_all_val_s
        all_val_loader = data_factory.get_dataloader(split=val_split, aug=None, shuffle=False,
                                                     target_list=range(end_class))
        all_val_loader_list.append(all_val_loader)

        # D_all_test_s
        all_test_loader = data_factory.get_dataloader(split=test_split, aug=None, shuffle=False,
                                                      target_list=range(end_class))
        all_test_loader_list.append(all_test_loader)

    # ----------------------
    # Model creation
    # ----------------------
    restune_model_dict = build_restune_model(args)

    # DEBUG
    # for ss in range(args.num_steps):
    #     print(restune_model_dict[f'step{ss}']['encoder'])
    #     print(restune_model_dict[f'step{ss}']['head_mix'])
    #     print(restune_model_dict[f'step{ss}']['head_res'])
    #     print('\n')

    print(args)

    if args.mode == 'train':
        sinkhorn = SinkhornKnopp(args)
        eval_results_recorder = StepResults(num_steps=args.num_steps)

        method = ResTune(
            # Model
            model_dict=restune_model_dict,
            # Sinkhorn cross-pseudo label generator
            sinkhorn=sinkhorn,
            # Evaluation results (test split) recorder
            eval_results_recorder=eval_results_recorder,
            # Train datasets
            step_train_loader_list=step_train_loader_list,
            # Val datasets
            step_val_loader_list=step_val_loader_list,
            prev_val_loader_list=prev_val_loader_list,
            all_val_loader_list=all_val_loader_list,
            # Test datasets
            step_test_loader_list=step_test_loader_list,
            prev_test_loader_list=prev_test_loader_list,
            all_test_loader_list=all_test_loader_list
        )

        # Step-wise Training
        for step_train in range(args.num_steps):
            # Init WandB logger
            wandb_run_name = f'ResTune_{args.dataset_name}_{args.model_name}_{step_train}/{args.num_steps}-Steps_Wkd_{args.w_kd}_GradBlock_{args.grad_from_block}'
            wandb_run_step = wandb.init(
                project='SOTA_ResTune',
                entity=args.wandb_entity,
                tags=[f'TotalStep={args.num_steps}', args.dataset_name, args.model_name, f'Wkd={args.w_kd}',
                      f'GradBlock={args.grad_from_block}', f'device={args.device}', f'CurrentStep={step_train}'],
                name=wandb_run_name,
                mode=args.wandb_mode,
                reinit=True
            )
            # Training
            #   1. Warm-up stage: only update classifier head
            method.warmup(args=args, step=step_train)
            #   2. Formal stage: unlock last block of ViT, update both encoder and classifier head
            method.train(args=args, step=step_train)

            # Testing
            #   1. Iterate test loaders
            method.test(args=args, step=step_train)
            #   2. Print test results
            method.show_eval_result(step=step_train)

            # Save old model for next incremental step
            method.duplicate_old_model(args, step=step_train)
            if step_train+1 < args.num_steps:
                wandb_run_step.finish()

        # Print final results
        for step in range(args.num_steps):
            method.show_eval_result(step=step)

        wandb_run_step.finish()
    elif args.mode == 'eval':
        raise NotImplementedError
    else:
        raise NotImplementedError
