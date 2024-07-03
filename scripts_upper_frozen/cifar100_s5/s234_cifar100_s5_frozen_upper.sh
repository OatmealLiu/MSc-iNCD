#!/bin/bash
#SBATCH -p gpupart
#SBATCH -A staff
#SBATCH --signal=B:SIGTERM@120
#SBATCH --gres gpu:1
#SBATCH --mem=32000

export PATH="/nfs/data_todi/mliu/software/anaconda3/bin:$PATH"

eval "$(conda shell.bash hook)"
bash

conda activate msc_incd

python -W ignore train_upper_joint_teacher_student.py \
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

python -W ignore train_upper_joint_teacher_student.py \
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

python -W ignore train_upper_joint_teacher_student.py \
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
