#!/bin/bash

python -W ignore train_waac.py \
        --epochs 1 \
        --batch_size 256 \
        --scale 1.0 \
        --normalization pcc \
        --softmax_temp 0.1\
        --dataset_root ./data/datasets/CIFAR/ \
        --dataset_name cifar100 \
        --num_classes 100 \
        --num_steps 5 \
        --current_step 0 \
        --exp_root ./outputs/ \
        --weights_root ./models/single_weights/ \
        --wandb_mode online \
        --wandb_entity oatmealliu
