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

python -W ignore train_upper_unlocked_joint_teacher_student.py \
        --epochs 100 \
        --batch_size 256 \
        --l2_single_cls \
        --dataset_root ./data/datasets/tiny-imagenet-200/ \
        --dataset_name tinyimagenet \
        --num_classes 200 \
        --num_steps 2 \
        --aug_type vit_uno \
        --current_step 1 \
        --stage stage1 \
        --mode train \
        --grad_from_block 11 \
        --num_mlp_layers 1 \
        --model_head LinearHead \
        --exp_root ./outputs/ \
        --weights_root ./models/single_weights/ \
        --exp_marker warmedup \
        --wandb_mode online