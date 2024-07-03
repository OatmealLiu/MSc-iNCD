#!/bin/bash
#SBATCH -p long
#SBATCH -A elisa.ricci
#SBATCH --gres gpu:1
#SBATCH --mem=32000
#SBATCH --time 48:00:00

export PATH="/home/mingxuan.liu/software/anaconda3/bin:$PATH"

eval "$(conda shell.bash hook)"
bash

conda activate msc_incd

python -W ignore train_upper_unlocked_joint_teacher_student.py \
        --epochs 100 \
        --batch_size 128 \
        --l2_single_cls \
        --dataset_root ./data/datasets/CIFAR/ \
        --dataset_name cifar10 \
        --num_classes 10 \
        --num_steps 2 \
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