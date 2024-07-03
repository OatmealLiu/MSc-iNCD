#!/bin/bash

DATASET_NAME="cub200"
TASKS=2

EPOCHS=1
BS=128
BUFFER_SIZE=50

python -W ignore train_il_der.py \
        --epochs_warmup $EPOCHS \
        --epochs $EPOCHS \
        --batch_size $BS \
        --alpha_der 0.5 \
        --buffer_size $BUFFER_SIZE \
        --dataset_name $DATASET_NAME \
        --num_steps $TASKS \
        --grad_from_block 11 \
        --wandb_mode offline \
        --wandb_entity oatmealliu
