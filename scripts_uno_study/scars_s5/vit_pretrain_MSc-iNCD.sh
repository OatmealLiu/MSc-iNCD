#!/bin/bash
#SBATCH -p long
#SBATCH -A elisa.ricci
#SBATCH --gres gpu:1
#SBATCH --mem=64000
#SBATCH --time 48:00:00

export PATH="/home/mingxuan.liu/software/anaconda3/bin:$PATH"

eval "$(conda shell.bash hook)"
bash

conda activate msc_incd

DATASET_NAME="scars"
ENCODER_NAME="vit_dino"
BS=256
TASKS=5

python -W ignore train_our_direct_concat.py \
        --epochs 200 \
        --batch_size $BS \
        --dataset_name $DATASET_NAME \
        --num_steps $TASKS \
        --model_name $ENCODER_NAME \
        --current_step 0 \
        --num_mlp_layers 1 \
        --dino_pretrain_path ./models/dino_weights/dino_vitbase16_pretrain.pth \
        --model_head LinearHead \
        --seed 10 \
        --exp_root ./outputs_uno_study/ \
        --weights_root ./models/single_weights_resnet50_dino/ \
        --wandb_mode online \
        --wandb_entity oatmealliu

python -W ignore train_our_direct_concat.py \
        --epochs 200 \
        --batch_size $BS \
        --dataset_name $DATASET_NAME \
        --num_steps $TASKS \
        --model_name $ENCODER_NAME \
        --current_step 1 \
        --num_mlp_layers 1 \
        --dino_pretrain_path ./models/dino_weights/dino_vitbase16_pretrain.pth \
        --model_head LinearHead \
        --seed 10 \
        --exp_root ./outputs_uno_study/ \
        --weights_root ./models/single_weights_resnet50_dino/ \
        --wandb_mode online \
        --wandb_entity oatmealliu

python -W ignore train_our_direct_concat.py \
        --epochs 200 \
        --batch_size $BS \
        --dataset_name $DATASET_NAME \
        --num_steps $TASKS \
        --model_name $ENCODER_NAME \
        --current_step 2 \
        --num_mlp_layers 1 \
        --dino_pretrain_path ./models/dino_weights/dino_vitbase16_pretrain.pth \
        --model_head LinearHead \
        --seed 10 \
        --exp_root ./outputs_uno_study/ \
        --weights_root ./models/single_weights_resnet50_dino/ \
        --wandb_mode online \
        --wandb_entity oatmealliu

python -W ignore train_our_direct_concat.py \
        --epochs 200 \
        --batch_size $BS \
        --dataset_name $DATASET_NAME \
        --num_steps $TASKS \
        --model_name $ENCODER_NAME \
        --current_step 3 \
        --num_mlp_layers 1 \
        --dino_pretrain_path ./models/dino_weights/dino_vitbase16_pretrain.pth \
        --model_head LinearHead \
        --seed 10 \
        --exp_root ./outputs_uno_study/ \
        --weights_root ./models/single_weights_resnet50_dino/ \
        --wandb_mode online \
        --wandb_entity oatmealliu

python -W ignore train_our_direct_concat.py \
        --epochs 200 \
        --batch_size $BS \
        --dataset_name $DATASET_NAME \
        --num_steps $TASKS \
        --model_name $ENCODER_NAME \
        --current_step 4 \
        --num_mlp_layers 1 \
        --dino_pretrain_path ./models/dino_weights/dino_vitbase16_pretrain.pth \
        --model_head LinearHead \
        --seed 10 \
        --exp_root ./outputs_uno_study/ \
        --weights_root ./models/single_weights_resnet50_dino/ \
        --wandb_mode online \
        --wandb_entity oatmealliu