#!/bin/bash

python -W ignore train_data_fading.py \
        --epochs 200 \
        --batch_size 256 \
        --percentage_filter 0.1 \
        --fading_step 50 \
        --dataset_root ./data/datasets/CIFAR/ \
        --dataset_name cifar100 \
        --num_classes 100 \
        --num_steps 5 \
        --current_step 1 \
        --exp_root ./outputs_data_fading/ \
        --wandb_mode offline \
        --wandb_entity oatmealliu