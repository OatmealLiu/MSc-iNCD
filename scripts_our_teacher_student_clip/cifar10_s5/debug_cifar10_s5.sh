#!/bin/bash

python -W ignore train_our_teacher_student.py \
        --epochs 1 \
        --batch_size 64 \
        --l2_single_cls \
        --student_loss ZP \
        --dataset_root ./data/datasets/CIFAR/ \
        --dataset_name cifar10 \
        --num_classes 10 \
        --aug_type vit_uno \
        --num_steps 5 \
        --current_step 1 \
        --mode train \
        --num_mlp_layers 1 \
        --model_head LinearHead \
        --exp_root ./outputs/ \
        --weights_root ./models/single_weights/ \
        --wandb_mode online