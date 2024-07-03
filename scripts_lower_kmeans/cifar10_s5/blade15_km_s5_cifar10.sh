#!/bin/bash

python -W ignore train_lower_kmeans.py \
        --dataset_root ./data/datasets/CIFAR/ \
        --dataset_name cifar10 \
        --num_classes 10 \
        --num_steps 5 \
        --km_max_iter 200 \
        --seed 10 \
        --wandb_mode online \
        --wandb_entity oatmealliu