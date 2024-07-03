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

DATASET_NAME="cub200"
ENCODER_NAME="vit_dino"
BS=256
TASKS=2
EPOCHS=200

for step in $(seq 0 1)
do
  python -W ignore train_our_direct_concat.py \
          --epochs $EPOCHS \
          --batch_size $BS \
          --dataset_name $DATASET_NAME \
          --num_steps $TASKS \
          --model_name $ENCODER_NAME \
          --current_step $step \
          --apply_l2weights \
          --num_mlp_layers 1 \
          --dino_pretrain_path ./models/dino_weights/dino_vitbase16_pretrain.pth \
          --model_head LinearHead \
          --seed 10 \
          --exp_root ./outputs_ours_wo_l2weights/ \
          --weights_root ./models/single_weights_wo_l2weights/ \
          --wandb_mode online \
          --wandb_entity oatmealliu
done
