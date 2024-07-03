#!/bin/bash

python -W ignore train_our_direct_concat.py \
        --epochs 1 \
        --batch_size 256 \
        --dataset_root ./data/datasets/CUB_200_2011/ \
        --dataset_name cub200 \
        --num_classes 200 \
        --num_steps 5 \
        --current_step 2 \
        --num_mlp_layers 1 \
        --dino_pretrain_path ./models/dino_weights/dino_vitbase16_pretrain.pth \
        --model_head LinearHead \
        --seed 10 \
        --exp_root ./outputs/ \
        --weights_root ./models/single_weights/ \
        --wandb_mode offline \
        --wandb_entity oatmealliu

