#!/bin/bash

num_gpus=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
echo "Number of GPUs available: $num_gpus"

torchrun --nnodes 1 --nproc_per_node $num_gpus run_dycoder.py configs/dycoder/gsm_dycoder.yaml
# python -m debugpy --listen 5678 --wait-for-client run_dycoder.py configs/dycoder/gsm_dycoder.yaml