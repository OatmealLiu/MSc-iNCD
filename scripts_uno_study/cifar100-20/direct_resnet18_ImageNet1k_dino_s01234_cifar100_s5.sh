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

python -W ignore train_our_direct_concat.py \
        --epochs 200 \
        --batch_size 256 \
        --dataset_root ./data/datasets/CIFAR/ \
        --dataset_name cifar100 \
        --num_classes 100 \
        --model_name resnet18_imagenet1k \
        --num_steps 5 \
        --current_step 0 \
        --num_mlp_layers 1 \
        --dino_pretrain_path ./models/dino_weights/dino_vitbase16_pretrain.pth \
        --model_head LinearHead \
        --seed 10 \
        --exp_root ./outputs_uno_study/ \
        --weights_root ./models/single_weights_resnet18_imagenet1k/ \
        --wandb_mode online \
        --wandb_entity oatmealliu

python -W ignore train_our_direct_concat.py \
        --epochs 200 \
        --batch_size 256 \
        --dataset_root ./data/datasets/CIFAR/ \
        --dataset_name cifar100 \
        --num_classes 100 \
        --model_name resnet18_imagenet1k \
        --num_steps 5 \
        --current_step 1 \
        --num_mlp_layers 1 \
        --dino_pretrain_path ./models/dino_weights/dino_vitbase16_pretrain.pth \
        --model_head LinearHead \
        --seed 10 \
        --exp_root ./outputs_uno_study/ \
        --weights_root ./models/single_weights_resnet18_imagenet1k/ \
        --wandb_mode online \
        --wandb_entity oatmealliu

python -W ignore train_our_direct_concat.py \
        --epochs 200 \
        --batch_size 256 \
        --dataset_root ./data/datasets/CIFAR/ \
        --dataset_name cifar100 \
        --num_classes 100 \
        --model_name resnet18_imagenet1k \
        --num_steps 5 \
        --current_step 2 \
        --num_mlp_layers 1 \
        --dino_pretrain_path ./models/dino_weights/dino_vitbase16_pretrain.pth \
        --model_head LinearHead \
        --seed 10 \
        --exp_root ./outputs_uno_study/ \
        --weights_root ./models/single_weights_resnet18_imagenet1k/ \
        --wandb_mode online \
        --wandb_entity oatmealliu

python -W ignore train_our_direct_concat.py \
        --epochs 200 \
        --batch_size 256 \
        --dataset_root ./data/datasets/CIFAR/ \
        --dataset_name cifar100 \
        --num_classes 100 \
        --model_name resnet18_imagenet1k \
        --num_steps 5 \
        --current_step 3 \
        --num_mlp_layers 1 \
        --dino_pretrain_path ./models/dino_weights/dino_vitbase16_pretrain.pth \
        --model_head LinearHead \
        --seed 10 \
        --exp_root ./outputs_uno_study/ \
        --weights_root ./models/single_weights_resnet18_imagenet1k/ \
        --wandb_mode online \
        --wandb_entity oatmealliu

python -W ignore train_our_direct_concat.py \
        --epochs 200 \
        --batch_size 256 \
        --dataset_root ./data/datasets/CIFAR/ \
        --dataset_name cifar100 \
        --num_classes 100 \
        --model_name resnet18_imagenet1k \
        --num_steps 5 \
        --current_step 4 \
        --num_mlp_layers 1 \
        --dino_pretrain_path ./models/dino_weights/dino_vitbase16_pretrain.pth \
        --model_head LinearHead \
        --seed 10 \
        --exp_root ./outputs_uno_study/ \
        --weights_root ./models/single_weights_resnet18_imagenet1k/ \
        --wandb_mode online \
        --wandb_entity oatmealliu