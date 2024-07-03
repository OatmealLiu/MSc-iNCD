#!/bin/bash

python -W ignore train_sota_rankstats.py \
        --epochs 10 \
        --batch_size 256 \
        --topk 5 \
        --w_bce 1.0 \
        --w_entropy 0.0 \
        --dataset_root ./data/datasets/tiny-imagenet-200/ \
        --dataset_name tinyimagenet \
        --num_classes 200 \
        --aug_type vit_uno \
        --num_steps 5 \
        --current_step 0 \
        --mode train \
        --exp_root ./outputs/ \
        --wandb_mode online
