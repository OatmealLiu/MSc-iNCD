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

DATASET_NAME="tinyimagenet"
TASKS=5

EPOCHS=35
BS=256
BUFFER_SIZE=200

python -W ignore train_il_der.py \
        --epochs_warmup $EPOCHS \
        --epochs $EPOCHS \
        --batch_size $BS \
        --alpha_der 0.5 \
        --buffer_size $BUFFER_SIZE \
        --dataset_name $DATASET_NAME \
        --num_steps $TASKS \
        --grad_from_block 11 \
        --wandb_mode online \
        --wandb_entity oatmealliu
