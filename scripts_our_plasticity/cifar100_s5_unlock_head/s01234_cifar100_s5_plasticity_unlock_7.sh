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

DATASET_NAME="cifar100"
ENCODER_NAME="vit_dino"
EPOCH=120
EPOCH_UNLOCK=60
BS=128
TASKS=5
GRAD_BLOCK=7

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
        --wandb_mode online \
        --wandb_entity oatmealliu

python -W ignore train_our_direct_concat_plasticity.py \
        --epochs $EPOCH \
        --unlock_epoch $EPOCH_UNLOCK \
        --batch_size $BS \
        --dataset_name $DATASET_NAME \
        --num_steps $TASKS \
        --model_name $ENCODER_NAME \
        --current_step 1 \
        --grad_from_block $GRAD_BLOCK \
        --num_mlp_layers 1 \
        --dino_pretrain_path ./models/dino_weights/dino_vitbase16_pretrain.pth \
        --model_head LinearHead \
        --seed 10 \
        --exp_root ./outputs_plasticity/ \
        --wandb_mode online \
        --wandb_entity oatmealliu

python -W ignore train_our_direct_concat_plasticity.py \
        --epochs $EPOCH \
        --unlock_epoch $EPOCH_UNLOCK \
        --batch_size $BS \
        --dataset_name $DATASET_NAME \
        --num_steps $TASKS \
        --model_name $ENCODER_NAME \
        --current_step 2 \
        --grad_from_block $GRAD_BLOCK \
        --num_mlp_layers 1 \
        --dino_pretrain_path ./models/dino_weights/dino_vitbase16_pretrain.pth \
        --model_head LinearHead \
        --seed 10 \
        --exp_root ./outputs_plasticity/ \
        --wandb_mode online \
        --wandb_entity oatmealliu

python -W ignore train_our_direct_concat_plasticity.py \
        --epochs $EPOCH \
        --unlock_epoch $EPOCH_UNLOCK \
        --batch_size $BS \
        --dataset_name $DATASET_NAME \
        --num_steps $TASKS \
        --model_name $ENCODER_NAME \
        --current_step 3 \
        --grad_from_block $GRAD_BLOCK \
        --num_mlp_layers 1 \
        --dino_pretrain_path ./models/dino_weights/dino_vitbase16_pretrain.pth \
        --model_head LinearHead \
        --seed 10 \
        --exp_root ./outputs_plasticity/ \
        --wandb_mode online \
        --wandb_entity oatmealliu

python -W ignore train_our_direct_concat_plasticity.py \
        --epochs $EPOCH \
        --unlock_epoch $EPOCH_UNLOCK \
        --batch_size $BS \
        --dataset_name $DATASET_NAME \
        --num_steps $TASKS \
        --model_name $ENCODER_NAME \
        --current_step 4 \
        --grad_from_block $GRAD_BLOCK \
        --num_mlp_layers 1 \
        --dino_pretrain_path ./models/dino_weights/dino_vitbase16_pretrain.pth \
        --model_head LinearHead \
        --seed 10 \
        --exp_root ./outputs_plasticity/ \
        --wandb_mode online \
        --wandb_entity oatmealliu
