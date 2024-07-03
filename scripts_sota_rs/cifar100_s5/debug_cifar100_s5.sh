#!/bin/bash

python -W ignore train_sota_rankstats.py \
        --epochs 1 \
        --batch_size 256 \
        --topk 5 \
        --w_bce 2.0 \
        --w_entropy 0.0 \
        --dataset_root ./data/datasets/CIFAR/ \
        --dataset_name cifar100 \
        --num_classes 100 \
        --aug_type vit_uno \
        --num_steps 5 \
        --current_step 1 \
        --mode train \
        --exp_root ./outputs/ \
        --wandb_mode offline

python -W ignore train_sota_rankstats.py \
        --epochs 1 \
        --batch_size 256 \
        --topk 5 \
        --w_bce 2.0 \
        --w_entropy 0.0 \
        --dataset_root ./data/datasets/CIFAR/ \
        --dataset_name cifar100 \
        --num_classes 100 \
        --aug_type vit_uno \
        --num_steps 5 \
        --current_step 2 \
        --mode train \
        --exp_root ./outputs/ \
        --wandb_mode offline

python -W ignore train_sota_rankstats.py \
        --epochs 1 \
        --batch_size 256 \
        --topk 5 \
        --w_bce 2.0 \
        --w_entropy 0.0 \
        --dataset_root ./data/datasets/CIFAR/ \
        --dataset_name cifar100 \
        --num_classes 100 \
        --aug_type vit_uno \
        --num_steps 5 \
        --current_step 3 \
        --mode train \
        --exp_root ./outputs/ \
        --wandb_mode offline

python -W ignore train_sota_rankstats.py \
        --epochs 1 \
        --batch_size 256 \
        --topk 5 \
        --w_bce 2.0 \
        --w_entropy 0.0 \
        --dataset_root ./data/datasets/CIFAR/ \
        --dataset_name cifar100 \
        --num_classes 100 \
        --aug_type vit_uno \
        --num_steps 5 \
        --current_step 4 \
        --mode train \
        --exp_root ./outputs/ \
        --wandb_mode offline
