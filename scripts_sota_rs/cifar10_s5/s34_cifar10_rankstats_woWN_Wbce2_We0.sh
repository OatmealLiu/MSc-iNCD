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

python -W ignore train_sota_rankstats.py \
        --epochs 100 \
        --batch_size 256 \
        --topk 5 \
        --w_bce 2.0 \
        --w_entropy 0.0 \
        --dataset_root ./data/datasets/CIFAR/ \
        --dataset_name cifar10 \
        --num_classes 10 \
        --aug_type vit_uno \
        --num_steps 5 \
        --current_step 3 \
        --mode train \
        --exp_root ./outputs/ \
        --wandb_mode online

python -W ignore train_sota_rankstats.py \
        --epochs 100 \
        --batch_size 256 \
        --topk 5 \
        --w_bce 2.0 \
        --w_entropy 0.0 \
        --dataset_root ./data/datasets/CIFAR/ \
        --dataset_name cifar10 \
        --num_classes 10 \
        --aug_type vit_uno \
        --num_steps 5 \
        --current_step 4 \
        --mode train \
        --exp_root ./outputs/ \
        --wandb_mode online