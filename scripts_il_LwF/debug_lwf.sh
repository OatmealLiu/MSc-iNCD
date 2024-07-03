#!/bin/bash

python -W ignore train_il_LwF.py \
        --epochs_warmup 1 \
        --epochs 1 \
        --batch_size 128 \
        --dataset_root ./data/datasets/CIFAR/ \
        --dataset_name cifar100 \
        --num_classes 100 \
        --num_steps 5 \
        --grad_from_block 11 \
        --wandb_mode online \
        --wandb_entity oatmealliu
