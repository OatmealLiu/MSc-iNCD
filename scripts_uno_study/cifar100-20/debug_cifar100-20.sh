#!/bin/bash

python -W ignore train_uno.py \
        --epochs_pretrain 1 \
        --epochs_ncd 1 \
        --batch_size 64 \
        --dataset_root ./data/datasets/CIFAR/ \
        --dataset_name cifar100 \
        --num_classes 100 \
        --num_base 80 \
        --num_novel 20 \
        --model_name resnet50_plain \
        --grad_from_block 11 \
        --exp_root ./outputs_uno_study/ \
        --wandb_mode offline \
        --wandb_entity oatmealliu
