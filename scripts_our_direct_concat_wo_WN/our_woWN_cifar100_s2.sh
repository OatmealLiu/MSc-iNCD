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

DATASET_NAME="cifar100"
ENCODER_NAME="vit_dino"
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
          --exp_root ./outputs_ours_wo_l2weights/ \
          --weights_root ./models/single_weights_wo_l2weights/ \
          --wandb_mode online \
          --wandb_entity oatmealliu
done
