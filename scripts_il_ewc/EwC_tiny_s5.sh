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

python -W ignore train_il_ewc.py \
        --epochs_warmup 100 \
        --epochs 100 \
        --batch_size 256 \
        --dataset_root ./data/datasets/tiny-imagenet-200/ \
        --dataset_name tinyimagenet \
        --num_classes 200 \
        --num_steps 5 \
        --grad_from_block 11 \
        --w_ewc 10000 \
        --alpha_ewc 0.5 \
        --wandb_mode online \
        --wandb_entity oatmealliu