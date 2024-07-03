#!/bin/bash


DATASET_NAME="aircraft"
ENCODER_NAME="resnet50_dino"
BS=32
TASKS=2

python -W ignore train_uno.py \
        --epochs_pretrain 2 \
        --epochs_ncd 2 \
        --batch_size $BS \
        --dataset_name $DATASET_NAME \
        --num_steps $TASKS \
        --model_name $ENCODER_NAME \
        --grad_from_block 11 \
        --exp_root ./outputs_uno_study/ \
        --wandb_mode offline \
        --wandb_entity oatmealliu