#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1

cd "$SLURM_SUBMIT_DIR"

# Array to store script names
script_name=(
    "eval_protomask.py"
)

home_dir=$(echo ~)

# Array to store script arguments
script_arguments=(

    "--experiment_sub_dir ProtoMask/cub200 --training_phase fine_tune --explanation_type prp --segmentation_method slit --crop --fill_background_method original --num_workers 32 --batch_size 16 --general --compactness --contrastivity --complexity"
    "--experiment_sub_dir ProtoMask/cub200 --training_phase fine_tune --explanation_type prp --segmentation_method sam2 --crop --fill_background_method original --num_workers 32 --batch_size 16 --general --compactness --contrastivity --complexity"

    "--experiment_sub_dir ProtoMask/cub200 --training_phase fine_tune --explanation_type prp --segmentation_method slit --crop --fill_background_method original --num_workers 32 --batch_size 16 --topk_prototype_images"
    "--experiment_sub_dir ProtoMask/cub200 --training_phase fine_tune --explanation_type prp --segmentation_method sam2 --crop --fill_background_method original --num_workers 32 --batch_size 16 --topk_prototype_images"

    "--experiment_sub_dir ProtoMask/dogs --training_phase fine_tune --explanation_type prp --segmentation_method slit --crop --fill_background_method original --num_workers 32 --batch_size 16 --general --compactness --contrastivity --complexity"
    "--experiment_sub_dir ProtoMask/dogs --training_phase fine_tune --explanation_type prp --segmentation_method sam2 --crop --fill_background_method original --num_workers 32 --batch_size 16 --general --compactness --contrastivity --complexity"

    "--experiment_sub_dir ProtoMask/dogs --training_phase fine_tune --explanation_type prp --segmentation_method slit --crop --fill_background_method original --num_workers 32 --batch_size 16 --topk_prototype_images"
    "--experiment_sub_dir ProtoMask/dogs --training_phase fine_tune --explanation_type prp --segmentation_method sam2 --crop --fill_background_method original --num_workers 32 --batch_size 16 --topk_prototype_images"

    "--experiment_sub_dir ProtoMask/cars196 --training_phase fine_tune --explanation_type prp --segmentation_method slit --crop --fill_background_method original --num_workers 32 --batch_size 16 --general --compactness --contrastivity --complexity"
    "--experiment_sub_dir ProtoMask/cars196 --training_phase fine_tune --explanation_type prp --segmentation_method sam2 --crop --fill_background_method original --num_workers 32 --batch_size 16 --general --compactness --contrastivity --complexity"

    "--experiment_sub_dir ProtoMask/cars196 --training_phase fine_tune --explanation_type prp --segmentation_method slit --crop --fill_background_method original --num_workers 32 --batch_size 16 --topk_prototype_images"
    "--experiment_sub_dir ProtoMask/cars196 --training_phase fine_tune --explanation_type prp --segmentation_method sam2 --crop --fill_background_method original --num_workers 32 --batch_size 16 --topk_prototype_images"
)

# Iterate over the scripts and run them
for ((i = 0; i < ${#script_arguments[@]}; i++)); do
    arguments="${script_arguments[$i]}"
    echo "Running $script_name with arguments: $arguments"
    python3 "$script_name" $arguments
done