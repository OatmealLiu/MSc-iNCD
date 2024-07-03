#!/bin/bash

python -W ignore eval_rankstats.py \
        --epochs 100 \
        --batch_size 256 \
        --use_norm \
        --topk 5 \
        --w_bce 3.0 \
        --w_entropy 1.0 \
        --dataset_root ./data/datasets/tiny-imagenet-200/ \
        --dataset_name tinyimagenet \
        --num_classes 200 \
        --aug_type vit_uno \
        --num_steps 2 \
        --current_step 0 \
        --mode train \
        --exp_root ./outputs/ \
        --wandb_mode offline

python -W ignore eval_rankstats.py \
        --epochs 100 \
        --batch_size 256 \
        --use_norm \
        --topk 5 \
        --w_bce 3.0 \
        --w_entropy 1.0 \
        --dataset_root ./data/datasets/tiny-imagenet-200/ \
        --dataset_name tinyimagenet \
        --num_classes 200 \
        --aug_type vit_uno \
        --num_steps 2 \
        --current_step 1 \
        --mode train \
        --exp_root ./outputs/ \
        --wandb_mode offline
