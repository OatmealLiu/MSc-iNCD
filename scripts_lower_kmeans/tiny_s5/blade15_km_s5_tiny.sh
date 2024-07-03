#!/bin/bash

python -W ignore train_lower_kmeans.py \
        --dataset_root ./data/datasets/tiny-imagenet-200/ \
        --dataset_name tinyimagenet \
        --num_classes 200 \
        --num_steps 5 \
        --km_max_iter 200 \
        --seed 10 \
        --wandb_mode online \
        --wandb_entity oatmealliu