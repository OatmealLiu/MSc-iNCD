#!/bin/bash

python -W ignore train_weight_discrepancy_nearest.py \
        --epochs 1 \
        --batch_size 256 \
        --w_wd rampup \
        --w_topk 2 \
        --epoch_wd 100 \
        --rampup_length 175 \
        --rampup_coefficient 25 \
        --dataset_root ./data/datasets/CIFAR/ \
        --dataset_name cifar100 \
        --num_classes 100 \
        --num_steps 5 \
        --current_step 0 \
        --exp_root ./outputs_weight_discrepancy/ \
        --wandb_mode offline \
        --wandb_entity oatmealliu

python -W ignore train_weight_discrepancy_nearest.py \
        --epochs 1 \
        --batch_size 256 \
        --w_wd rampup \
        --w_topk 2 \
        --epoch_wd 100 \
        --rampup_length 175 \
        --rampup_coefficient 25 \
        --dataset_root ./data/datasets/CIFAR/ \
        --dataset_name cifar100 \
        --num_classes 100 \
        --num_steps 5 \
        --current_step 1 \
        --exp_root ./outputs_weight_discrepancy/ \
        --wandb_mode offline \
        --wandb_entity oatmealliu

python -W ignore train_weight_discrepancy_nearest.py \
        --epochs 1 \
        --batch_size 256 \
        --w_wd rampup \
        --w_topk 2 \
        --epoch_wd 100 \
        --rampup_length 175 \
        --rampup_coefficient 25 \
        --dataset_root ./data/datasets/CIFAR/ \
        --dataset_name cifar100 \
        --num_classes 100 \
        --num_steps 5 \
        --current_step 2 \
        --exp_root ./outputs_weight_discrepancy/ \
        --wandb_mode offline \
        --wandb_entity oatmealliu

python -W ignore train_weight_discrepancy_nearest.py \
        --epochs 1 \
        --batch_size 256 \
        --w_wd rampup \
        --w_topk 2 \
        --epoch_wd 100 \
        --rampup_length 175 \
        --rampup_coefficient 25 \
        --dataset_root ./data/datasets/CIFAR/ \
        --dataset_name cifar100 \
        --num_classes 100 \
        --num_steps 5 \
        --current_step 4 \
        --exp_root ./outputs_weight_discrepancy/ \
        --wandb_mode offline \
        --wandb_entity oatmealliu

python -W ignore train_weight_discrepancy_nearest.py \
        --epochs 1 \
        --batch_size 256 \
        --w_wd rampup \
        --w_topk 2 \
        --epoch_wd 100 \
        --rampup_length 175 \
        --rampup_coefficient 25 \
        --dataset_root ./data/datasets/CIFAR/ \
        --dataset_name cifar100 \
        --num_classes 100 \
        --num_steps 5 \
        --current_step 5 \
        --exp_root ./outputs_weight_discrepancy/ \
        --wandb_mode offline \
        --wandb_entity oatmealliu