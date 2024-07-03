#!/bin/bash

python -W ignore train_manifold_hopper.py \
        --epochs 1 \
        --batch_size 256 \
        --dim_reduction 256 \
        --pred_method voting \
        --feat_slice cutoff \
        --dataset_root ./data/datasets/CIFAR/ \
        --dataset_name cifar100 \
        --num_classes 100 \
        --num_steps 5 \
        --current_step 1 \
        --exp_root ./outputs_manifold_hopper/ \
        --wandb_mode offline \
        --wandb_entity oatmealliu