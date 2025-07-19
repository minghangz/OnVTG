#!/bin/bash

if [ -z "$CUDA_VISIBLE_DEVICES" ]; then
  export CUDA_VISIBLE_DEVICES=0
fi

IFS=',' read -ra GPU_LIST <<< "$CUDA_VISIBLE_DEVICES"
NUM_GPUS=${#GPU_LIST[@]}

MASTER_ADDR=localhost
MASTER_PORT=$((12000 + RANDOM % 20000))

CONFIG=$1

torchrun --nproc_per_node=$NUM_GPUS --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT \
train.py \
  config="$CONFIG" \
  "${@:2}"
