#!/bin/bash

python -W ignore train_sota_rankstats.py \
        --epochs 5 \
        --batch_size 256 \
        --topk 5 \
        --w_bce 2.0 \
        --w_entropy 0.0 \
        --softmax_temp 1.0 \
        --dataset_root ./data/datasets/CIFAR/ \
        --dataset_name cifar10 \
        --num_classes 10 \
        --aug_type vit_uno \
        --num_steps 5 \
        --current_step 0 \
        --mode train \
        --exp_root ./outputs/ \
        --wandb_mode offline