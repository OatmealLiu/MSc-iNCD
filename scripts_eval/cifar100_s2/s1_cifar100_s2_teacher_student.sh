#!/bin/bash

python -W ignore eval_direct.py \
        --epochs 200 \
        --batch_size 256 \
        --l2_single_cls \
        --student_loss ZP \
        --dataset_root ./data/datasets/CIFAR/ \
        --dataset_name cifar100 \
        --num_classes 100 \
        --aug_type vit_uno \
        --num_steps 2 \
        --current_step 1 \
        --mode train \
        --num_mlp_layers 1 \
        --model_head LinearHead \
        --exp_root ./outputs/ \
        --weights_root ./models/single_weights/ \
        --wandb_mode offline