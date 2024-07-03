#!/bin/bash
#SBATCH -p long
#SBATCH -A elisa.ricci
#SBATCH --gres gpu:2
#SBATCH --mem=64000
#SBATCH --time 48:00:00

export PATH="/home/mingxuan.liu/software/anaconda3/bin:$PATH"

eval "$(conda shell.bash hook)"
bash

conda activate msc_incd

python -W ignore train_uno.py \
        --epochs_pretrain 200 \
        --epochs_ncd 200 \
        --batch_size 256 \
        --dataset_root ./data/datasets/CIFAR/ \
        --dataset_name cifar100 \
        --num_classes 100 \
        --num_base 80 \
        --num_novel 20 \
        --model_name resnet18_imagenet1k \
        --grad_from_block 11 \
        --exp_root ./outputs_uno_study/ \
        --wandb_mode online \
        --wandb_entity oatmealliu