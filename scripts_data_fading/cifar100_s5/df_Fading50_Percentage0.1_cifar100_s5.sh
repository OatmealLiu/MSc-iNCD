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

python -W ignore train_data_fading.py \
        --epochs 200 \
        --batch_size 256 \
        --percentage_filter 0.1 \
        --fading_step 50 \
        --dataset_root ./data/datasets/CIFAR/ \
        --dataset_name cifar100 \
        --num_classes 100 \
        --num_steps 5 \
        --current_step 4 \
        --exp_root ./outputs_data_fading/ \
        --wandb_mode online \
        --wandb_entity oatmealliu