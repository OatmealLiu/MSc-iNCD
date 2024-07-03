#!/bin/bash

python -W ignore train_uno.py \
        --epochs 1 \
        --batch_size 32 \
        --dataset_root ./data/datasets/CIFAR/ \
        --dataset_name cifar10 \
        --num_classes 10 \
        --num_base 5 \
        --num_novel 5 \
        --model_name vit_dino \
        --grad_from_block 11 \
        --exp_root ./outputs_uno_study/ \
        --wandb_mode offline \
        --wandb_entity oatmealliu