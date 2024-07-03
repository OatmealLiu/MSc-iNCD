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

python -W ignore train_il_frost.py \
        --epochs 200 \
        --batch_size 256 \
        --step_size 170 \
        --rampup_length 150 \
        --rampup_coefficient 25 \
        --aug_type vit_uno \
        --wandb_mode online \
        --exp_root ./outputs_frost/ \
        --grad_from_block 11 \
        --dataset_root ./data/datasets/CIFAR/ \
        --dataset_name cifar100 \
        --num_classes 100 \
        --num_steps 5 \
        --current_step 0

python -W ignore train_il_frost.py \
        --epochs 200 \
        --batch_size 256 \
        --step_size 170 \
        --rampup_length 150 \
        --rampup_coefficient 25 \
        --aug_type vit_uno \
        --wandb_mode online \
        --exp_root ./outputs_frost/ \
        --grad_from_block 11 \
        --dataset_root ./data/datasets/CIFAR/ \
        --dataset_name cifar100 \
        --num_classes 100 \
        --num_steps 5 \
        --current_step 1

python -W ignore train_il_frost.py \
        --epochs 200 \
        --batch_size 256 \
        --step_size 170 \
        --rampup_length 150 \
        --rampup_coefficient 25 \
        --aug_type vit_uno \
        --wandb_mode online \
        --exp_root ./outputs_frost/ \
        --grad_from_block 11 \
        --dataset_root ./data/datasets/CIFAR/ \
        --dataset_name cifar100 \
        --num_classes 100 \
        --num_steps 5 \
        --current_step 2

python -W ignore train_il_frost.py \
        --epochs 200 \
        --batch_size 256 \
        --step_size 170 \
        --rampup_length 150 \
        --rampup_coefficient 25 \
        --aug_type vit_uno \
        --wandb_mode online \
        --exp_root ./outputs_frost/ \
        --grad_from_block 11 \
        --dataset_root ./data/datasets/CIFAR/ \
        --dataset_name cifar100 \
        --num_classes 100 \
        --num_steps 5 \
        --current_step 3

python -W ignore train_il_frost.py \
        --epochs 200 \
        --batch_size 256 \
        --step_size 170 \
        --rampup_length 150 \
        --rampup_coefficient 25 \
        --aug_type vit_uno \
        --wandb_mode online \
        --exp_root ./outputs_frost/ \
        --grad_from_block 11 \
        --dataset_root ./data/datasets/CIFAR/ \
        --dataset_name cifar100 \
        --num_classes 100 \
        --num_steps 5 \
        --current_step 4