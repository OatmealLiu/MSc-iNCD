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

python -W ignore train_sota_scan.py \
        --epochs_scan 100 \
        --epochs_selflabel 100 \
        --batch_size 256 \
        --use_norm \
        --apply_class_balancing \
        --dataset_root ./data/datasets/CIFAR/ \
        --dataset_name cifar100 \
        --num_classes 100 \
        --num_steps 5 \
        --current_step 0 \
        --exp_root ./outputs_scan \
        --wandb_mode online \
        --wandb_entity oatmealliu

python -W ignore train_sota_scan.py \
        --epochs_scan 100 \
        --epochs_selflabel 100 \
        --batch_size 256 \
        --use_norm \
        --apply_class_balancing \
        --dataset_root ./data/datasets/CIFAR/ \
        --dataset_name cifar100 \
        --num_classes 100 \
        --num_steps 5 \
        --current_step 1 \
        --exp_root ./outputs_scan \
        --wandb_mode online \
        --wandb_entity oatmealliu

python -W ignore train_sota_scan.py \
        --epochs_scan 100 \
        --epochs_selflabel 100 \
        --batch_size 256 \
        --use_norm \
        --apply_class_balancing \
        --dataset_root ./data/datasets/CIFAR/ \
        --dataset_name cifar100 \
        --num_classes 100 \
        --num_steps 5 \
        --current_step 2 \
        --exp_root ./outputs_scan \
        --wandb_mode online \
        --wandb_entity oatmealliu

python -W ignore train_sota_scan.py \
        --epochs_scan 100 \
        --epochs_selflabel 100 \
        --batch_size 256 \
        --use_norm \
        --apply_class_balancing \
        --dataset_root ./data/datasets/CIFAR/ \
        --dataset_name cifar100 \
        --num_classes 100 \
        --num_steps 5 \
        --current_step 3 \
        --exp_root ./outputs_scan \
        --wandb_mode online \
        --wandb_entity oatmealliu

python -W ignore train_sota_scan.py \
        --epochs_scan 100 \
        --epochs_selflabel 100 \
        --batch_size 256 \
        --use_norm \
        --apply_class_balancing \
        --dataset_root ./data/datasets/CIFAR/ \
        --dataset_name cifar100 \
        --num_classes 100 \
        --num_steps 5 \
        --current_step 4 \
        --exp_root ./outputs_scan \
        --wandb_mode online \
        --wandb_entity oatmealliu
