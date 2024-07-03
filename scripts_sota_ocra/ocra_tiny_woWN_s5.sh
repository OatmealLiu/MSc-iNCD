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

DATASET_NAME="cifar10"
ENCODER_NAME="vit_dino"
BS=256
TASKS=5
EPOCHS=100

for step in $(seq 0 4)
do
  python -W ignore train_sota_ocra.py \
          --epochs $EPOCHS \
          --batch_size $BS \
          --topk 2 \
          --w_bce 1.0 \
          --w_entropy 1.0 \
          --dataset_name $DATASET_NAME \
          --aug_type vit_uno \
          --num_steps $TASKS \
          --current_step $step \
          --mode train \
          --exp_root ./outputs_ocra \
          --wandb_mode online \
          --wandb_entity oatmealliu
done