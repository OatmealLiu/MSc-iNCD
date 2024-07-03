#!/bin/bash

DATASET_NAME="cub200"
BS=32
TASKS=2
EPOCHS=1

python -W ignore train_upper_unlocked_joint_teacher_student.py \
        --epochs $EPOCHS \
        --batch_size $BS \
        --dataset_name $DATASET_NAME \
        --num_steps $TASKS \
        --aug_type vit_uno \
        --current_step 0 \
        --stage stage1 \
        --mode train \
        --grad_from_block 11 \
        --num_mlp_layers 1 \
        --model_head LinearHead \
        --exp_root ./outputs_upper_unlocked/ \
        --weights_root ./models/single_weights/ \
        --exp_marker warmedup \
        --wandb_mode offline

python -W ignore train_upper_unlocked_joint_teacher_student.py \
        --epochs $EPOCHS \
        --batch_size $BS \
        --dataset_name $DATASET_NAME \
        --num_steps $TASKS \
        --aug_type vit_uno \
        --current_step 1 \
        --stage stage1 \
        --mode train \
        --grad_from_block 11 \
        --num_mlp_layers 1 \
        --model_head LinearHead \
        --exp_root ./outputs_upper_unlocked/ \
        --weights_root ./models/single_weights/ \
        --exp_marker warmedup \
        --wandb_mode offline

python -W ignore train_upper_unlocked_joint_teacher_student.py \
        --epochs $EPOCHS \
        --batch_size $BS \
        --dataset_name $DATASET_NAME \
        --num_steps $TASKS \
        --aug_type vit_uno \
        --current_step 1 \
        --stage stage2 \
        --mode train \
        --grad_from_block 11 \
        --num_mlp_layers 1 \
        --model_head LinearHead \
        --exp_root ./outputs_upper_unlocked/ \
        --weights_root ./models/single_weights/ \
        --exp_marker warmedup \
        --wandb_mode offline