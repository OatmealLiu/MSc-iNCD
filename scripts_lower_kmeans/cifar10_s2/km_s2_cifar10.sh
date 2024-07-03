#!/bin/bash

#SBATCH -A IscrC_MC-iNCD
#SBATCH -p dgx_usr_preempt
#SBATCH -q normal
#SBATCH --time 24:00:00               # format: HH:MM:SS
#SBATCH -N 1                          # 1 node
#SBATCH --ntasks-per-node=8          # 8 tasks
#SBATCH --gres=gpu:1                  # 1 gpus per node out of 8
#SBATCH --mem=64GB                    # memory per node out of 980000 MB

export PATH="/dgx/home/userexternal/mliu0000/miniconda3/bin:$PATH"

eval "$(conda shell.bash hook)"
bash

conda activate base

DATASET_NAME="cifar10"
TASKS=2
KM_ITER=200

python -W ignore train_lower_kmeans.py \
        --dataset_name $DATASET_NAME \
        --num_steps $TASKS \
        --km_max_iter $KM_ITER \
        --seed 10 \
        --wandb_mode online \
        --wandb_entity oatmealliu