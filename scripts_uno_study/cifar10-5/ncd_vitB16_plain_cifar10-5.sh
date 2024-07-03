#!/bin/bash
#SBATCH -A IscrC_MC-iNCD
#SBATCH -p dgx_usr_prod
#SBATCH -q dgx_qos_sprod
#SBATCH --time 48:00:00               # format: HH:MM:SS
#SBATCH -N 1                          # 1 node
#SBATCH --ntasks-per-node=8          # 8 tasks
#SBATCH --gres=gpu:2                  # 1 gpus per node out of 8
#SBATCH --mem=245000MB                    # memory per node out of 980000 MB

export PATH="/dgx/home/userexternal/mliu0000/miniconda3/bin:$PATH"

eval "$(conda shell.bash hook)"
bash

conda activate base

DATASET_NAME="cifar10"
ENCODER_NAME="vit_plain"
BS=64
TASKS=2

python -W ignore train_uno.py \
        --epochs_pretrain 200 \
        --epochs_ncd 100 \
        --batch_size $BS \
        --dataset_name $DATASET_NAME \
        --num_steps $TASKS \
        --model_name $ENCODER_NAME \
        --grad_from_block 11 \
        --exp_root ./outputs_uno_study/ \
        --wandb_mode online \
        --wandb_entity oatmealliu