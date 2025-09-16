#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=128
#SBATCH --gres=gpu:1

cd "$SLURM_SUBMIT_DIR"

# Array to store script names
script_name=(
    "lightning_protopool.py"
)

home_dir=$(echo ~)
# Array to store script arguments
script_arguments=(
    "--dataset cub200 --method sam2 --seed 42 --n_trials 25"
)

# Iterate over the scripts and run them
for ((i = 0; i < ${#script_arguments[@]}; i++)); do
    arguments="${script_arguments[$i]}"
    echo "Running $script_name with arguments: $arguments"
    python3 "$script_name" $arguments
done
