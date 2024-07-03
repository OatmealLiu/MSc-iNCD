#!/bin/bash
#SBATCH -p long
#SBATCH -A elisa.ricci
#SBATCH --gres gpu:1
#SBATCH --mem=64000
#SBATCH --time 48:00:00

export PATH="/home/mingxuan.liu/software/anaconda3/bin:$PATH"

eval "$(conda shell.bash hook)"
bash

conda activate msc_incd

DATASET_NAME="tinyimagenet"
ENCODER_NAME="resnet50_dino"
BS=256
TASKS=2
EPOCHS=100

for step in $(seq 0 1)
do
  python -W ignore train_our_direct_concat.py \
          --epochs $EPOCHS \
          --batch_size $BS \
          --dataset_name $DATASET_NAME \
          --num_steps $TASKS \
          --model_name $ENCODER_NAME \
          --current_step $step \
          --num_mlp_layers 1 \
          --dino_pretrain_path ./models/dino_weights/dino_vitbase16_pretrain.pth \
          --model_head LinearHead \
          --seed 10 \
          --exp_root ./outputs_uno_study/ \
          --weights_root ./models/single_weights_resnet50_dino/ \
          --wandb_mode online \
          --wandb_entity oatmealliu
done
