#!/bin/bash

DATASET_NAME="cifar100"
ENCODER_NAME="vit_dino"
EPOCH=2
EPOCH_UNLOCK=0
BS=32
TASKS=5
GRAD_BLOCK=8

python -W ignore train_our_direct_concat_plasticity.py \
        --epochs $EPOCH \
        --unlock_epoch $EPOCH_UNLOCK \
        --batch_size $BS \
        --dataset_name $DATASET_NAME \
        --num_steps $TASKS \
        --model_name $ENCODER_NAME \
        --current_step 0 \
        --grad_from_block $GRAD_BLOCK \
        --num_mlp_layers 1 \
        --dino_pretrain_path ./models/dino_weights/dino_vitbase16_pretrain.pth \
        --model_head LinearHead \
        --seed 10 \
        --exp_root ./outputs_plasticity/ \
        --wandb_mode offline \
        --wandb_entity oatmealliu
