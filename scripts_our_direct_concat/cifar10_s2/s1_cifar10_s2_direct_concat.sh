#!/bin/bash
#SBATCH -p gpupart
#SBATCH -A staff
#SBATCH --signal=B:SIGTERM@120
#SBATCH --gres gpu:1
#SBATCH --mem=32000

export PATH="/nfs/data_todi/mliu/software/anaconda3/bin:$PATH"

eval "$(conda shell.bash hook)"
bash

conda activate msc_incd

python -W ignore train_our_direct_concat.py \
        --epochs 100 \
        --batch_size 256 \
        --dataset_root ./data/datasets/CIFAR/ \
        --dataset_name cifar10 \
        --num_classes 10 \
        --num_steps 2 \
        --current_step 1 \
        --num_mlp_layers 1 \
        --dino_pretrain_path ./models/dino_weights/dino_vitbase16_pretrain.pth \
        --model_head LinearHead \
        --seed 10 \
        --exp_root ./outputs/ \
        --weights_root ./models/single_weights/ \
        --wandb_mode online \
        --wandb_entity oatmealliu