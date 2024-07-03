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

python -W ignore train_our_teacher_student.py \
        --epochs 100 \
        --batch_size 256 \
        --l2_single_cls \
        --student_loss ZP \
        --dataset_root ./data/datasets/tiny-imagenet-200/ \
        --dataset_name tinyimagenet \
        --num_classes 200 \
        --aug_type vit_uno_clip \
        --model_name clip \
        --num_steps 5 \
        --current_step 1 \
        --mode train \
        --num_mlp_layers 1 \
        --model_head LinearHead \
        --exp_root ./outputs_clip/ \
        --weights_root ./models/single_weights_clip/ \
        --wandb_mode online

python -W ignore train_our_teacher_student.py \
        --epochs 100 \
        --batch_size 256 \
        --l2_single_cls \
        --student_loss ZP \
        --dataset_root ./data/datasets/tiny-imagenet-200/ \
        --dataset_name tinyimagenet \
        --num_classes 200 \
        --aug_type vit_uno_clip \
        --model_name clip \
        --num_steps 5 \
        --current_step 2 \
        --mode train \
        --num_mlp_layers 1 \
        --model_head LinearHead \
        --exp_root ./outputs_clip/ \
        --weights_root ./models/single_weights_clip/ \
        --wandb_mode online

python -W ignore train_our_teacher_student.py \
        --epochs 100 \
        --batch_size 256 \
        --l2_single_cls \
        --student_loss ZP \
        --dataset_root ./data/datasets/tiny-imagenet-200/ \
        --dataset_name tinyimagenet \
        --num_classes 200 \
        --aug_type vit_uno_clip \
        --model_name clip \
        --num_steps 5 \
        --current_step 3 \
        --mode train \
        --num_mlp_layers 1 \
        --model_head LinearHead \
        --exp_root ./outputs_clip/ \
        --weights_root ./models/single_weights_clip/ \
        --wandb_mode online

python -W ignore train_our_teacher_student.py \
        --epochs 100 \
        --batch_size 256 \
        --l2_single_cls \
        --student_loss ZP \
        --dataset_root ./data/datasets/tiny-imagenet-200/ \
        --dataset_name tinyimagenet \
        --num_classes 200 \
        --aug_type vit_uno_clip \
        --model_name clip \
        --num_steps 5 \
        --current_step 4 \
        --mode train \
        --num_mlp_layers 1 \
        --model_head LinearHead \
        --exp_root ./outputs_clip/ \
        --weights_root ./models/single_weights_clip/ \
        --wandb_mode online