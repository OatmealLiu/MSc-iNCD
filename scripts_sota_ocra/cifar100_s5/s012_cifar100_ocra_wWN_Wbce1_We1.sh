#!/bin/bash
#SBATCH -p chaos
#SBATCH -A shared-mhug-staff
#SBATCH --signal=B:SIGTERM@120
#SBATCH --gres gpu:1
#SBATCH --mem=32000

export PATH="/nfs/data_chaos/mliu/software/anaconda3/bin:$PATH"

eval "$(conda shell.bash hook)"
bash

conda activate msc_incd

python -W ignore train_sota_ocra.py \
        --epochs 100 \
        --batch_size 256 \
        --use_norm \
        --topk 2 \
        --w_bce 1.0 \
        --w_entropy 1.0 \
        --dataset_root ./data/datasets/CIFAR/ \
        --dataset_name cifar100 \
        --num_classes 100 \
        --aug_type vit_uno \
        --num_steps 5 \
        --current_step 0 \
        --mode train \
        --exp_root ./outputs/ \
        --wandb_mode online

python -W ignore train_sota_ocra.py \
        --epochs 100 \
        --batch_size 256 \
        --use_norm \
        --topk 2 \
        --w_bce 1.0 \
        --w_entropy 1.0 \
        --dataset_root ./data/datasets/CIFAR/ \
        --dataset_name cifar100 \
        --num_classes 100 \
        --aug_type vit_uno \
        --num_steps 5 \
        --current_step 1 \
        --mode train \
        --exp_root ./outputs/ \
        --wandb_mode online

python -W ignore train_sota_ocra.py \
        --epochs 100 \
        --batch_size 256 \
        --use_norm \
        --topk 2 \
        --w_bce 1.0 \
        --w_entropy 1.0 \
        --dataset_root ./data/datasets/CIFAR/ \
        --dataset_name cifar100 \
        --num_classes 100 \
        --aug_type vit_uno \
        --num_steps 5 \
        --current_step 2 \
        --mode train \
        --exp_root ./outputs/ \
        --wandb_mode online