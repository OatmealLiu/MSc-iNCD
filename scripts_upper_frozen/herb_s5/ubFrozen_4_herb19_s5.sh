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

DATASET_NAME="herb19"
BS=256
TASKS=5
EPOCHS=100
step=4

python -W ignore train_upper_joint_teacher_student.py \
        --epochs $EPOCHS \
        --batch_size $BS \
        --student_loss ZP \
        --dataset_name $DATASET_NAME \
        --aug_type vit_uno \
        --num_steps $TASKS \
        --current_step $step \
        --mode train \
        --num_mlp_layers 1 \
        --model_head LinearHead \
        --exp_root ./outputs_upper_frozen/ \
        --weights_root ./models/single_weights/ \
        --wandb_mode online