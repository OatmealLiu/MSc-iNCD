#!/bin/bash
#SBATCH -A IscrC_MC-iNCD
#SBATCH -p dgx_usr_prod
#SBATCH -q dgx_qos_sprod
#SBATCH --time 48:00:00               # format: HH:MM:SS
#SBATCH -N 1                          # 1 node
#SBATCH --ntasks-per-node=8          # 8 tasks
#SBATCH --gres=gpu:2                  # 1 gpus per node out of 8
#SBATCH --mem=64GB                    # memory per node out of 980000 MB

export PATH="/dgx/home/userexternal/mliu0000/miniconda3/bin:$PATH"

eval "$(conda shell.bash hook)"
bash

conda activate base

python -W ignore train_uno.py \
        --epochs_pretrain 200 \
        --epochs_ncd 200 \
        --batch_size 256 \
        --dataset_root ./data/datasets/CIFAR/ \
        --dataset_name cifar10 \
        --num_classes 10 \
        --num_base 5 \
        --num_novel 5 \
        --model_name resnet18_plain \
        --grad_from_block 11 \
        --exp_root ./outputs_uno_study/ \
        --wandb_mode online \
        --wandb_entity oatmealliu