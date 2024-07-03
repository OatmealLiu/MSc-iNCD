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

DATASET_NAME="cifar100"
ENCODER_NAME="resnet50_dino"
BS=256
TASKS=5
EPOCHS=200
GRAD_BLOCK=11

for step in $(seq 0 4)
do
  python -W ignore train_ncd_incremental.py \
          --epochs $EPOCHS \
          --batch_size $BS \
          --dataset_name $DATASET_NAME \
          --num_steps $TASKS \
          --model_name $ENCODER_NAME \
          --grad_from_block $GRAD_BLOCK \
          --current_step $step \
          --num_mlp_layers 1 \
          --dino_pretrain_path ./models/dino_weights/dino_vitbase16_pretrain.pth \
          --model_head LinearHead \
          --seed 10 \
          --exp_root ./outputs_ncdil_study/ \
          --weights_root ./models/single_weights_resnet50_dino/ \
          --wandb_mode online \
          --wandb_entity oatmealliu
done