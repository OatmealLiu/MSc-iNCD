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

python -W ignore train_manifold_hopper.py \
        --epochs 200 \
        --batch_size 256 \
        --dim_reduction 256 \
        --pred_method average \
        --feat_slice cutoff \
        --dataset_root ./data/datasets/CIFAR/ \
        --dataset_name cifar100 \
        --num_classes 100 \
        --num_steps 5 \
        --current_step 0 \
        --exp_root ./outputs_manifold_hopper/ \
        --wandb_mode online \
        --wandb_entity oatmealliu

python -W ignore train_manifold_hopper.py \
        --epochs 200 \
        --batch_size 256 \
        --dim_reduction 256 \
        --pred_method average \
        --feat_slice cutoff \
        --dataset_root ./data/datasets/CIFAR/ \
        --dataset_name cifar100 \
        --num_classes 100 \
        --num_steps 5 \
        --current_step 1 \
        --exp_root ./outputs_manifold_hopper/ \
        --wandb_mode online \
        --wandb_entity oatmealliu

python -W ignore train_manifold_hopper.py \
        --epochs 200 \
        --batch_size 256 \
        --dim_reduction 256 \
        --pred_method average \
        --feat_slice cutoff \
        --dataset_root ./data/datasets/CIFAR/ \
        --dataset_name cifar100 \
        --num_classes 100 \
        --num_steps 5 \
        --current_step 2 \
        --exp_root ./outputs_manifold_hopper/ \
        --wandb_mode online \
        --wandb_entity oatmealliu

python -W ignore train_manifold_hopper.py \
        --epochs 200 \
        --batch_size 256 \
        --dim_reduction 256 \
        --pred_method average \
        --feat_slice cutoff \
        --dataset_root ./data/datasets/CIFAR/ \
        --dataset_name cifar100 \
        --num_classes 100 \
        --num_steps 5 \
        --current_step 3 \
        --exp_root ./outputs_manifold_hopper/ \
        --wandb_mode online \
        --wandb_entity oatmealliu

python -W ignore train_manifold_hopper.py \
        --epochs 200 \
        --batch_size 256 \
        --dim_reduction 256 \
        --pred_method average \
        --feat_slice cutoff \
        --dataset_root ./data/datasets/CIFAR/ \
        --dataset_name cifar100 \
        --num_classes 100 \
        --num_steps 5 \
        --current_step 4 \
        --exp_root ./outputs_manifold_hopper/ \
        --wandb_mode online \
        --wandb_entity oatmealliu