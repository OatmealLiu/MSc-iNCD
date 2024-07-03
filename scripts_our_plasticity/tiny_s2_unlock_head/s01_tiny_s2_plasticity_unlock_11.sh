#!/bin/bash
#SBATCH -p long
#SBATCH -A elisa.ricci
#SBATCH --gres gpu:1
#SBATCH --mem=32000
#SBATCH --time 48:00:00

export PATH="/home/mingxuan.liu/software/anaconda3/bin:$PATH"

eval "$(conda shell.bash hook)"
bash

conda activate msc_incd

python -W ignore train_our_direct_concat_plasticity.py \
        --epochs 200 \
        --unlock_epoch 100 \
        --batch_size 256 \
        --dataset_root ./data/datasets/tiny-imagenet-200/ \
        --dataset_name tinyimagenet \
        --num_classes 200 \
        --num_steps 2 \
        --current_step 0 \
        --grad_from_block 11 \
        --num_mlp_layers 1 \
        --dino_pretrain_path ./models/dino_weights/dino_vitbase16_pretrain.pth \
        --model_head LinearHead \
        --seed 10 \
        --exp_root ./outputs_plasticity/ \
        --wandb_mode online \
        --wandb_entity oatmealliu

python -W ignore train_our_direct_concat_plasticity.py \
        --epochs 200 \
        --unlock_epoch 100 \
        --batch_size 256 \
        --dataset_root ./data/datasets/tiny-imagenet-200/ \
        --dataset_name tinyimagenet \
        --num_classes 200 \
        --num_steps 2 \
        --current_step 1 \
        --grad_from_block 11 \
        --num_mlp_layers 1 \
        --dino_pretrain_path ./models/dino_weights/dino_vitbase16_pretrain.pth \
        --model_head LinearHead \
        --seed 10 \
        --exp_root ./outputs_plasticity/ \
        --wandb_mode online \
        --wandb_entity oatmealliu
