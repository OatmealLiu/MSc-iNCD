#!/bin/bash

python -W ignore train_weight_discrepancy.py \
        --epochs 2 \
        --batch_size 128 \
        --w_wd 0.01 \
        --epoch_wd 1 \
        --conf_guided \
        --dataset_root ./data/datasets/CIFAR/ \
        --dataset_name cifar100 \
        --num_classes 100 \
        --num_steps 5 \
        --current_step 1 \
        --exp_root ./outputs_weight_discrepancy/ \
        --wandb_mode offline \
        --wandb_entity oatmealliu