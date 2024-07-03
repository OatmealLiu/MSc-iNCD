#!/bin/bash

python -W ignore train_uno.py \
        --epochs_pretrain 1 \
        --epochs_ncd 1 \
        --batch_size 8 \
        --dataset_root ./data/datasets/CIFAR/ \
        --dataset_name cifar100 \
        --num_classes 100 \
        --num_base 50 \
        --num_novel 50 \
        --model_name resnet50_plain \
        --grad_from_block 11 \
        --exp_root ./outputs_uno_study/ \
        --wandb_mode online \
        --wandb_entity oatmealliu