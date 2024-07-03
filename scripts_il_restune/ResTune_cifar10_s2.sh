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

python -W ignore train_il_ResTune.py \
        --epochs_warmup 100 \
        --epochs 100 \
        --batch_size 64 \
        --dataset_name cifar10 \
        --num_steps 2 \
        --grad_from_block 10 \
        --wandb_mode online \
        --wandb_entity oatmealliu
