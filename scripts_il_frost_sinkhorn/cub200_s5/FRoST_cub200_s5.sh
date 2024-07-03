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

DATASET_NAME="cub200"
BS=256
TASKS=5
EPOCHS=200

for step in $(seq 0 4)
do
  python -W ignore train_il_frost.py \
          --labeling_method sinkhorn \
          --epochs $EPOCHS \
          --batch_size $BS \
          --step_size 170 \
          --rampup_length 50 \
          --rampup_coefficient 5.0 \
          --aug_type vit_uno \
          --wandb_mode online \
          --exp_root ./outputs_frost/ \
          --grad_from_block 11 \
          --dataset_name $DATASET_NAME \
          --num_steps $TASKS \
          --current_step $step
done