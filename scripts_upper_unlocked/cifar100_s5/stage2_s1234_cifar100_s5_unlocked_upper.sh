#!/bin/bash
#SBATCH -A IscrC_MC-iNCD
#SBATCH -p dgx_usr_prod
#SBATCH -q dgx_qos_sprod
#SBATCH --time 48:00:00               # format: HH:MM:SS
#SBATCH -N 1                          # 1 node
#SBATCH --ntasks-per-node=8          # 8 tasks
#SBATCH --gres=gpu:1                  # 1 gpus per node out of 8
#SBATCH --mem=64GB                    # memory per node out of 980000 MB

export PATH="/dgx/home/userexternal/mliu0000/miniconda3/bin:$PATH"

eval "$(conda shell.bash hook)"
bash

conda activate base

python -W ignore train_upper_unlocked_joint_teacher_student.py \
        --epochs 100 \
        --batch_size 128 \
        --l2_single_cls \
        --dataset_root ./data/datasets/CIFAR/ \
        --dataset_name cifar100 \
        --num_classes 100 \
        --num_steps 5 \
        --aug_type vit_uno \
        --current_step 1 \
        --stage stage2 \
        --mode train \
        --grad_from_block 11 \
        --num_mlp_layers 1 \
        --model_head LinearHead \
        --exp_root ./outputs/ \
        --weights_root ./models/single_weights/ \
        --exp_marker warmedup \
        --wandb_mode online

python -W ignore train_upper_unlocked_joint_teacher_student.py \
        --epochs 100 \
        --batch_size 128 \
        --l2_single_cls \
        --dataset_root ./data/datasets/CIFAR/ \
        --dataset_name cifar100 \
        --num_classes 100 \
        --num_steps 5 \
        --aug_type vit_uno \
        --current_step 2 \
        --stage stage2 \
        --mode train \
        --grad_from_block 11 \
        --num_mlp_layers 1 \
        --model_head LinearHead \
        --exp_root ./outputs/ \
        --weights_root ./models/single_weights/ \
        --exp_marker warmedup \
        --wandb_mode online

python -W ignore train_upper_unlocked_joint_teacher_student.py \
        --epochs 100 \
        --batch_size 128 \
        --l2_single_cls \
        --dataset_root ./data/datasets/CIFAR/ \
        --dataset_name cifar100 \
        --num_classes 100 \
        --num_steps 5 \
        --aug_type vit_uno \
        --current_step 3 \
        --stage stage2 \
        --mode train \
        --grad_from_block 11 \
        --num_mlp_layers 1 \
        --model_head LinearHead \
        --exp_root ./outputs/ \
        --weights_root ./models/single_weights/ \
        --exp_marker warmedup \
        --wandb_mode online

python -W ignore train_upper_unlocked_joint_teacher_student.py \
        --epochs 100 \
        --batch_size 128 \
        --l2_single_cls \
        --dataset_root ./data/datasets/CIFAR/ \
        --dataset_name cifar100 \
        --num_classes 100 \
        --num_steps 5 \
        --aug_type vit_uno \
        --current_step 4 \
        --stage stage2 \
        --mode train \
        --grad_from_block 11 \
        --num_mlp_layers 1 \
        --model_head LinearHead \
        --exp_root ./outputs/ \
        --weights_root ./models/single_weights/ \
        --exp_marker warmedup \
        --wandb_mode online