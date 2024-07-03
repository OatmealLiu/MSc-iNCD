#!/bin/bash

DATASET_NAME="cifar100"
ENCODER_NAME="vit_dino"
BS=64
TASKS=5
EPOCHS=1

for step in $(seq 0 4)
do
  python -W ignore train_ncd_incremental.py \
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