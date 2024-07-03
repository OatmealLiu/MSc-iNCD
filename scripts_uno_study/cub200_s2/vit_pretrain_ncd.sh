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

DATASET_NAME="cub200"
ENCODER_NAME="vit_dino"
BS=256
TASKS=2

python -W ignore train_uno.py \
        --epochs_pretrain 200 \
        --epochs_ncd 100 \
        --batch_size $BS \
        --dataset_name $DATASET_NAME \
        --num_steps $TASKS \
        --model_name $ENCODER_NAME \
        --grad_from_block 11 \
        --exp_root ./outputs_uno_study/ \
        --wandb_mode online \
        --wandb_entity oatmealliu