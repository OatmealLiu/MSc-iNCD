#!/bin/bash

DATASET_NAME="cub200"
BS=64
TASKS=2
EPOCHS=1

for step in $(seq 1 TASKS-1)
do
  echo $step
done
