#!/bin/bash

#SBATCH -A IscrC_MC-iNCD
#SBATCH -p dgx_usr_preempt
#SBATCH -q normal
#SBATCH --time 24:00:00               # format: HH:MM:SS
#SBATCH -N 1                          # 1 node
#SBATCH --ntasks-per-node=8          # 8 tasks
#SBATCH --gres=gpu:1                  # 1 gpus per node out of 8
#SBATCH --mem=32GB                    # memory per node out of 980000 MB

export PATH="/dgx/home/userexternal/mliu0000/miniconda3/bin:$PATH"

eval "$(conda shell.bash hook)"
bash

conda activate base

DATASET_NAME="tinyimagenet"
TASKS=2

EPOCHS=40
BS=256
BUFFER_SIZE=200

python -W ignore train_il_der.py \
        --epochs_warmup $EPOCHS \
        --epochs $EPOCHS \
        --batch_size $BS \
        --alpha_der 0.5 \
        --buffer_size $BUFFER_SIZE \
        --dataset_name $DATASET_NAME \
        --num_steps $TASKS \
        --grad_from_block 11 \
        --wandb_mode online \
        --wandb_entity oatmealliu
