#!/bin/bash
#SBATCH --job-name=SLIT_Dogs
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=128
#SBATCH --gres=gpu:1

cd "$SLURM_SUBMIT_DIR"

# Array to store script names
script_name=(
    "lightning_protomask.py"
)

home_dir=$(echo ~)
# Array to store script arguments
script_arguments=(
    "--features resnet50 --dataset cars196 --fold_idx 0 --crop_input  --fill_background_method original --segmentation_method sam2 --num_workers 32 --tune --n_trials 50"
    "--features resnet50 --dataset cub200 --fold_idx 0 --crop_input  --fill_background_method original --segmentation_method sam2 --num_workers 32 --tune --n_trials 50"
    "--features resnet50 --dataset dogs --fold_idx 0 --crop_input  --fill_background_method original --segmentation_method sam2 --num_workers 32 --tune --n_trials 50"

    "--features resnet50 --dataset cars196 --fold_idx 0 --crop_input  --fill_background_method original --segmentation_method slit --num_workers 32 --tune --n_trials 50"
    "--features resnet50 --dataset cub200 --fold_idx 0 --crop_input  --fill_background_method original --segmentation_method slit --num_workers 32 --tune --n_trials 50"
    "--features resnet50 --dataset dogs --fold_idx 0 --crop_input  --fill_background_method original --segmentation_method slit --num_workers 32 --tune --n_trials 50"
)

# Iterate over the scripts and run them
for ((i = 0; i < ${#script_arguments[@]}; i++)); do
    arguments="${script_arguments[$i]}"
    echo "Running $script_name with arguments: $arguments"
    python3 "$script_name" $arguments
done
