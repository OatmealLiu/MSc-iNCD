#!/bin/bash

python -W ignore train_il_frost.py \
        --labeling_method sinkhorn \
        --epochs 1 \
        --batch_size 128 \
        --step_size 170 \
        --rampup_length 150 \
        --rampup_coefficient 25 \
        --aug_type vit_uno \
        --wandb_mode offline \
        --exp_root ./outputs_frost/ \
        --grad_from_block 11 \
        --dataset_root ./data/datasets/CIFAR/ \
        --dataset_name cifar100 \
        --num_classes 100 \
        --num_steps 5 \
        --current_step 0

python -W ignore train_il_frost.py \
        --labeling_method sinkhorn \
        --epochs 1 \
        --batch_size 128 \
        --step_size 170 \
        --rampup_length 150 \
        --rampup_coefficient 25 \
        --aug_type vit_uno \
        --wandb_mode offline \
        --exp_root ./outputs_frost/ \
        --grad_from_block 11 \
        --dataset_root ./data/datasets/CIFAR/ \
        --dataset_name cifar100 \
        --num_classes 100 \
        --num_steps 5 \
        --current_step 1

python -W ignore train_il_frost.py \
        --labeling_method sinkhorn \
        --epochs 1 \
        --batch_size 128 \
        --step_size 170 \
        --rampup_length 150 \
        --rampup_coefficient 25 \
        --aug_type vit_uno \
        --wandb_mode offline \
        --exp_root ./outputs_frost/ \
        --grad_from_block 11 \
        --dataset_root ./data/datasets/CIFAR/ \
        --dataset_name cifar100 \
        --num_classes 100 \
        --num_steps 5 \
        --current_step 2

python -W ignore train_il_frost.py \
        --labeling_method sinkhorn \
        --epochs 1 \
        --batch_size 128 \
        --step_size 170 \
        --rampup_length 150 \
        --rampup_coefficient 25 \
        --aug_type vit_uno \
        --wandb_mode offline \
        --exp_root ./outputs_frost/ \
        --grad_from_block 11 \
        --dataset_root ./data/datasets/CIFAR/ \
        --dataset_name cifar100 \
        --num_classes 100 \
        --num_steps 5 \
        --current_step 3

python -W ignore train_il_frost.py \
        --labeling_method sinkhorn \
        --epochs 1 \
        --batch_size 128 \
        --step_size 170 \
        --rampup_length 150 \
        --rampup_coefficient 25 \
        --aug_type vit_uno \
        --wandb_mode offline \
        --exp_root ./outputs_frost/ \
        --grad_from_block 11 \
        --dataset_root ./data/datasets/CIFAR/ \
        --dataset_name cifar100 \
        --num_classes 100 \
        --num_steps 5 \
        --current_step 4

