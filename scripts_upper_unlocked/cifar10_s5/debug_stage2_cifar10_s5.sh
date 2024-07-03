#!/bin/bash

python -W ignore train_upper_unlocked_joint_teacher_student.py \
        --epochs 1 \
        --batch_size 8 \
        --l2_single_cls \
        --dataset_root ./data/datasets/CIFAR/ \
        --dataset_name cifar10 \
        --num_classes 10 \
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