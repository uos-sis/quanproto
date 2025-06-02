#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=128
#SBATCH --gres=gpu:1

cd "$SLURM_SUBMIT_DIR"

# Array to store script names
script_name=(
    "eval_model.py"
)

home_dir=$(echo ~)

# Array to store script arguments
script_arguments=(

    "--experiment_sub_dir ProtoPNet/cub200  --dataset_dir $home_dir/data/quanproto --use_bbox --training_phase fine_tune --explanation_type prp --crop --num_workers 16 --general --compactness --contrastivity --continuity --complexity --output_completeness"
    "--experiment_sub_dir ProtoPNet/awa2    --dataset_dir $home_dir/data/quanproto --use_bbox --training_phase fine_tune --explanation_type prp --crop --num_workers 16 --multi_label --general --compactness --contrastivity --continuity"
    "--experiment_sub_dir ProtoPNet/cars196 --dataset_dir $home_dir/data/quanproto --use_bbox --training_phase fine_tune --explanation_type prp --crop --num_workers 16 --general --compactness --contrastivity --continuity"
    "--experiment_sub_dir ProtoPNet/nico    --dataset_dir $home_dir/data/quanproto --use_bbox --training_phase fine_tune --explanation_type prp --crop --num_workers 16 --general --compactness --contrastivity --continuity"

    "--experiment_sub_dir ProtoPNetPruned/cub200  --dataset_dir $home_dir/data/quanproto --use_bbox --training_phase fine_tune --explanation_type prp --crop --num_workers 16 --general --compactness --contrastivity --continuity --complexity --output_completeness"
    "--experiment_sub_dir ProtoPNetPruned/awa2    --dataset_dir $home_dir/data/quanproto --use_bbox --training_phase fine_tune --explanation_type prp --crop --num_workers 16 --multi_label --general --compactness --contrastivity --continuity"
    "--experiment_sub_dir ProtoPNetPruned/cars196 --dataset_dir $home_dir/data/quanproto --use_bbox --training_phase fine_tune --explanation_type prp --crop --num_workers 16 --general --compactness --contrastivity --continuity"
    "--experiment_sub_dir ProtoPNetPruned/nico    --dataset_dir $home_dir/data/quanproto --use_bbox --training_phase fine_tune --explanation_type prp --crop --num_workers 16 --general --compactness --contrastivity --continuity"

    "--experiment_sub_dir ProtoPool/cub200  --dataset_dir $home_dir/data/quanproto --use_bbox --training_phase fine_tune --explanation_type prp --crop --num_workers 16 --general --compactness --contrastivity --continuity --complexity --output_completeness"
    "--experiment_sub_dir ProtoPool/awa2    --dataset_dir $home_dir/data/quanproto --use_bbox --training_phase fine_tune --explanation_type prp --crop --num_workers 16 --multi_label --general --compactness --contrastivity --continuity"
    "--experiment_sub_dir ProtoPool/cars196 --dataset_dir $home_dir/data/quanproto --use_bbox --training_phase fine_tune --explanation_type prp --crop --num_workers 16 --general --compactness --contrastivity --continuity"
    "--experiment_sub_dir ProtoPool/nico    --dataset_dir $home_dir/data/quanproto --use_bbox --training_phase fine_tune --explanation_type prp --crop --num_workers 16 --general --compactness --contrastivity --continuity"

    "--experiment_sub_dir PIPNet/cub200  --dataset_dir $home_dir/data/quanproto --use_bbox --training_phase joint --explanation_type prp --crop --num_workers 16 --general --compactness --contrastivity --continuity --complexity --output_completeness"
    "--experiment_sub_dir PIPNet/awa2    --dataset_dir $home_dir/data/quanproto --use_bbox --training_phase joint --explanation_type prp --crop --num_workers 16 --multi_label --general --compactness --contrastivity --continuity"
    "--experiment_sub_dir PIPNet/cars196 --dataset_dir $home_dir/data/quanproto --use_bbox --training_phase joint --explanation_type prp --crop --num_workers 16 --general --compactness --contrastivity --continuity"
    "--experiment_sub_dir PIPNet/nico    --dataset_dir $home_dir/data/quanproto --use_bbox --training_phase joint --explanation_type prp --crop --num_workers 16 --general --compactness --contrastivity --continuity"
)

# Iterate over the scripts and run them
for ((i = 0; i < ${#script_arguments[@]}; i++)); do
    arguments="${script_arguments[$i]}"
    echo "Running $script_name with arguments: $arguments"
    python3 "$script_name" $arguments
done
