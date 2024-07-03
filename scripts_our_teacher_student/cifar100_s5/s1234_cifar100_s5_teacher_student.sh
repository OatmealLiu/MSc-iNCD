#!/bin/bash
#SBATCH -A IscrC_MC-iNCD
#SBATCH -p dgx_usr_preempt
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

python -W ignore train_our_teacher_student.py \
        --epochs 100 \
        --batch_size 256 \
        --l2_single_cls \
        --student_loss ZP \
        --dataset_root ./data/datasets/CIFAR/ \
        --dataset_name cifar100 \
        --num_classes 100 \
        --aug_type vit_uno \
        --num_steps 5 \
        --current_step 1 \
        --mode train \
        --num_mlp_layers 1 \
        --model_head LinearHead \
        --exp_root ./outputs/ \
        --weights_root ./models/single_weights/ \
        --wandb_mode online

python -W ignore train_our_teacher_student.py \
        --epochs 100 \
        --batch_size 256 \
        --l2_single_cls \
        --student_loss ZP \
        --dataset_root ./data/datasets/CIFAR/ \
        --dataset_name cifar100 \
        --num_classes 100 \
        --aug_type vit_uno \
        --num_steps 5 \
        --current_step 2 \
        --mode train \
        --num_mlp_layers 1 \
        --model_head LinearHead \
        --exp_root ./outputs/ \
        --weights_root ./models/single_weights/ \
        --wandb_mode online

python -W ignore train_our_teacher_student.py \
        --epochs 100 \
        --batch_size 256 \
        --l2_single_cls \
        --student_loss ZP \
        --dataset_root ./data/datasets/CIFAR/ \
        --dataset_name cifar100 \
        --num_classes 100 \
        --aug_type vit_uno \
        --num_steps 5 \
        --current_step 3 \
        --mode train \
        --num_mlp_layers 1 \
        --model_head LinearHead \
        --exp_root ./outputs/ \
        --weights_root ./models/single_weights/ \
        --wandb_mode online

python -W ignore train_our_teacher_student.py \
        --epochs 100 \
        --batch_size 256 \
        --l2_single_cls \
        --student_loss ZP \
        --dataset_root ./data/datasets/CIFAR/ \
        --dataset_name cifar100 \
        --num_classes 100 \
        --aug_type vit_uno \
        --num_steps 5 \
        --current_step 4 \
        --mode train \
        --num_mlp_layers 1 \
        --model_head LinearHead \
        --exp_root ./outputs/ \
        --weights_root ./models/single_weights/ \
        --wandb_mode online