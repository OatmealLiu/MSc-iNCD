#!/bin/bash

python -W ignore train_our_direct_concat.py \
        --epochs 1 \
        --batch_size 256 \
        --dataset_root ./data/datasets/herbarium_19/ \
        --dataset_name herb19 \
        --num_classes 683 \
        --num_steps 5 \
        --current_step 0 \
        --num_mlp_layers 1 \
        --dino_pretrain_path ./models/dino_weights/dino_vitbase16_pretrain.pth \
        --model_head LinearHead \
        --seed 10 \
        --exp_root ./outputs/ \
        --weights_root ./models/single_weights/ \
        --wandb_mode offline \
        --wandb_entity oatmealliu

python -W ignore train_our_direct_concat.py \
        --epochs 1 \
        --batch_size 256 \
        --dataset_root ./data/datasets/herbarium_19/ \
        --dataset_name herb19 \
        --num_classes 683 \
        --num_steps 5 \
        --current_step 1 \
        --num_mlp_layers 1 \
        --dino_pretrain_path ./models/dino_weights/dino_vitbase16_pretrain.pth \
        --model_head LinearHead \
        --seed 10 \
        --exp_root ./outputs/ \
        --weights_root ./models/single_weights/ \
        --wandb_mode offline \
        --wandb_entity oatmealliu

python -W ignore train_our_direct_concat.py \
        --epochs 1 \
        --batch_size 256 \
        --dataset_root ./data/datasets/herbarium_19/ \
        --dataset_name herb19 \
        --num_classes 683 \
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

python -W ignore train_our_direct_concat.py \
        --epochs 1 \
        --batch_size 256 \
        --dataset_root ./data/datasets/herbarium_19/ \
        --dataset_name herb19 \
        --num_classes 683 \
        --num_steps 5 \
        --current_step 3 \
        --num_mlp_layers 1 \
        --dino_pretrain_path ./models/dino_weights/dino_vitbase16_pretrain.pth \
        --model_head LinearHead \
        --seed 10 \
        --exp_root ./outputs/ \
        --weights_root ./models/single_weights/ \
        --wandb_mode offline \
        --wandb_entity oatmealliu

python -W ignore train_our_direct_concat.py \
        --epochs 1 \
        --batch_size 256 \
        --dataset_root ./data/datasets/herbarium_19/ \
        --dataset_name herb19 \
        --num_classes 683 \
        --num_steps 5 \
        --current_step 4 \
        --num_mlp_layers 1 \
        --dino_pretrain_path ./models/dino_weights/dino_vitbase16_pretrain.pth \
        --model_head LinearHead \
        --seed 10 \
        --exp_root ./outputs/ \
        --weights_root ./models/single_weights/ \
        --wandb_mode offline \
        --wandb_entity oatmealliu
