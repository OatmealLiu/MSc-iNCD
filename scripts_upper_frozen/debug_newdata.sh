#!/bin/bash

DATASET_NAME="cub200"
BS=64
TASKS=2
EPOCHS=1
step=1

python -W ignore train_upper_joint_teacher_student.py \
        --epochs $EPOCHS \
        --batch_size $BS \
        --student_loss ZP \
        --dataset_name $DATASET_NAME \
        --aug_type vit_uno \
        --num_steps $TASKS \
        --current_step $step \
        --mode train \
        --num_mlp_layers 1 \
        --model_head LinearHead \
        --exp_root ./outputs_upper_frozen/ \
        --weights_root ./models/single_weights/ \
        --wandb_mode offline