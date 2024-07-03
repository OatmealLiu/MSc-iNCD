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

DATASET_NAME="herb19"
TASKS=2

EPOCHS=100
BS=256

python -W ignore train_il_LwF.py \
        --epochs_warmup $EPOCHS \
        --epochs $EPOCHS \
        --batch_size $BS \
        --dataset_name $DATASET_NAME \
        --num_steps $TASKS \
        --grad_from_block 11 \
        --wandb_mode online \
        --wandb_entity oatmealliu
