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

DATASET_NAME="herb19"
BS=256
TASKS=5
EPOCHS=200

for step in $(seq 0 4)
do
  python -W ignore train_il_frost.py \
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