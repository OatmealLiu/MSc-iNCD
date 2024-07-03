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

python -W ignore train_sota_rankstats.py \
        --epochs 100 \
        --batch_size 256 \
        --rampup_length 50 \
        --rampup_coefficient 5.0 \
        --use_norm \
        --topk 5 \
        --w_bce 3.0 \
        --w_entropy 1.0 \
        --dataset_root ./data/datasets/CIFAR/ \
        --dataset_name cifar10 \
        --num_classes 10 \
        --aug_type vit_uno \
        --num_steps 2 \
        --current_step 0 \
        --mode train \
        --exp_root ./outputs/ \
        --wandb_mode online

python -W ignore train_sota_rankstats.py \
        --epochs 100 \
        --batch_size 256 \
        --rampup_length 50 \
        --rampup_coefficient 5.0 \
        --use_norm \
        --topk 5 \
        --w_bce 3.0 \
        --w_entropy 1.0 \
        --dataset_root ./data/datasets/CIFAR/ \
        --dataset_name cifar10 \
        --num_classes 10 \
        --aug_type vit_uno \
        --num_steps 2 \
        --current_step 1 \
        --mode train \
        --exp_root ./outputs/ \
        --wandb_mode online
