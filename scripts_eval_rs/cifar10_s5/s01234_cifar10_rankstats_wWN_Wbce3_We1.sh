#!/bin/bash

python -W ignore eval_rankstats.py \
        --epochs 200 \
        --batch_size 256 \
        --use_norm \
        --topk 5 \
        --w_bce 3.0 \
        --w_entropy 1.0 \
        --dataset_root ./data/datasets/CIFAR/ \
        --dataset_name cifar10 \
        --num_classes 10 \
        --aug_type vit_uno \
        --num_steps 5 \
        --current_step 0 \
        --mode train \
        --exp_root ./outputs/ \
        --wandb_mode offline

python -W ignore eval_rankstats.py \
        --epochs 200 \
        --batch_size 256 \
        --use_norm \
        --topk 5 \
        --w_bce 3.0 \
        --w_entropy 1.0 \
        --dataset_root ./data/datasets/CIFAR/ \
        --dataset_name cifar10 \
        --num_classes 10 \
        --aug_type vit_uno \
        --num_steps 5 \
        --current_step 1 \
        --mode train \
        --exp_root ./outputs/ \
        --wandb_mode offline

python -W ignore eval_rankstats.py \
        --epochs 200 \
        --batch_size 256 \
        --use_norm \
        --topk 5 \
        --w_bce 3.0 \
        --w_entropy 1.0 \
        --dataset_root ./data/datasets/CIFAR/ \
        --dataset_name cifar10 \
        --num_classes 10 \
        --aug_type vit_uno \
        --num_steps 5 \
        --current_step 2 \
        --mode train \
        --exp_root ./outputs/ \
        --wandb_mode offline

python -W ignore eval_rankstats.py \
        --epochs 200 \
        --batch_size 256 \
        --use_norm \
        --topk 5 \
        --w_bce 3.0 \
        --w_entropy 1.0 \
        --dataset_root ./data/datasets/CIFAR/ \
        --dataset_name cifar10 \
        --num_classes 10 \
        --aug_type vit_uno \
        --num_steps 5 \
        --current_step 3 \
        --mode train \
        --exp_root ./outputs/ \
        --wandb_mode offline

python -W ignore eval_rankstats.py \
        --epochs 200 \
        --batch_size 256 \
        --use_norm \
        --topk 5 \
        --w_bce 3.0 \
        --w_entropy 1.0 \
        --dataset_root ./data/datasets/CIFAR/ \
        --dataset_name cifar10 \
        --num_classes 10 \
        --aug_type vit_uno \
        --num_steps 5 \
        --current_step 4 \
        --mode train \
        --exp_root ./outputs/ \
        --wandb_mode offline
