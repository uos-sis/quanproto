#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=128
#SBATCH --gres=gpu:1

cd "$SLURM_SUBMIT_DIR"

# Array to store script names
script_name=(
    "lightning_protopnet.py"
)

home_dir=$(echo ~)
# Array to store script arguments
script_arguments=(
    "--features resnet50 --dataset cub200 --fold_idx 0 --crop_input --num_workers 16 --seed 42 --no_progress_bar"
    "--features resnet50 --dataset cub200 --fold_idx 1 --crop_input --num_workers 16 --seed 42 --no_progress_bar"
    "--features resnet50 --dataset cub200 --fold_idx 2 --crop_input --num_workers 16 --seed 42 --no_progress_bar"
    "--features resnet50 --dataset cub200 --fold_idx 3 --crop_input --num_workers 16 --seed 42 --no_progress_bar"
    
    "--features resnet50 --dataset cars196 --fold_idx 0 --crop_input --num_workers 16 --seed 42 --no_progress_bar"
    "--features resnet50 --dataset cars196 --fold_idx 1 --crop_input --num_workers 16 --seed 42 --no_progress_bar"
    "--features resnet50 --dataset cars196 --fold_idx 2 --crop_input --num_workers 16 --seed 42 --no_progress_bar"
    "--features resnet50 --dataset cars196 --fold_idx 3 --crop_input --num_workers 16 --seed 42 --no_progress_bar"

    "--features resnet50 --dataset nico --fold_idx 0 --crop_input --num_workers 16 --seed 42 --no_progress_bar"
    "--features resnet50 --dataset nico --fold_idx 1 --crop_input --num_workers 16 --seed 42 --no_progress_bar"
    "--features resnet50 --dataset nico --fold_idx 2 --crop_input --num_workers 16 --seed 42 --no_progress_bar"
    "--features resnet50 --dataset nico --fold_idx 3 --crop_input --num_workers 16 --seed 42 --no_progress_bar"
    
    "--features resnet50 --dataset awa2 --fold_idx 0 --crop_input --num_workers 16 --seed 42 --no_progress_bar"
    "--features resnet50 --dataset awa2 --fold_idx 1 --crop_input --num_workers 16 --seed 42 --no_progress_bar"
    "--features resnet50 --dataset awa2 --fold_idx 2 --crop_input --num_workers 16 --seed 42 --no_progress_bar"
    "--features resnet50 --dataset awa2 --fold_idx 3 --crop_input --num_workers 16 --seed 42 --no_progress_bar"
)

# Iterate over the scripts and run them
for ((i = 0; i < ${#script_arguments[@]}; i++)); do
    arguments="${script_arguments[$i]}"
    echo "Running $script_name with arguments: $arguments"
    python3 "$script_name" $arguments
done
