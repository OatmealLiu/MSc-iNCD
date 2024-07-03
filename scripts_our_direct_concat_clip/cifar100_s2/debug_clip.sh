#!/bin/bash

python -W ignore train_our_direct_concat.py \
        --epochs 1 \
        --batch_size 256 \
        --dataset_root ./data/datasets/CIFAR/ \
        --dataset_name cifar100 \
        --num_classes 100 \
        --aug_type vit_uno_clip \
        --model_name clip \
        --num_steps 2 \
        --current_step 1 \
        --num_mlp_layers 1 \
        --dino_pretrain_path ./models/dino_weights/dino_vitbase16_pretrain.pth \
        --model_head LinearHead \
        --seed 10 \
        --exp_root ./outputs_clip/ \
        --weights_root ./models/single_weights_clip/ \
        --wandb_mode offline \
        --wandb_entity oatmealliu