#!/bin/bash

# Array to store script names
script_name=(
    "lightning_pipnet.py"
)

home_dir=$(echo ~)
# Array to store script arguments
script_arguments=(
    "--features resnet50 --dataset_dir $home_dir/data/quanproto --dataset cub200 --fold_idx 0 --crop_input --num_workers 16 --seed 42 --tune --n_trials 100"
)

# Iterate over the scripts and run them
for ((i = 0; i < ${#script_arguments[@]}; i++)); do
    arguments="${script_arguments[$i]}"
    echo "Running $script_name with arguments: $arguments"
    python3 "$script_name" $arguments
done
