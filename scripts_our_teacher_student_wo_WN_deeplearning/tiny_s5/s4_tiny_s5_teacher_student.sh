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

DATASET_NAME="tinyimagenet"
BS=256
TASKS=5
EPOCHS=200
step=4

python -W ignore train_our_teacher_student.py \
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
        --exp_root ./outputs_teacher_student_wo_l2weights/ \
        --weights_root ./models/single_weights_wo_l2weights/ \
        --wandb_mode online
