#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1

cd "$SLURM_SUBMIT_DIR"

# Array to store script names
script_name=(
    "eval_model.py"
)

home_dir=$(echo ~)

# Array to store script arguments
script_arguments=(
    # PIPNet
    "--experiment_sub_dir PIPNet/cub200 --use_bbox --crop --training_phase joint --explanation_type prp --num_workers 16 --general --compactness --contrastivity --complexity"
    "--experiment_sub_dir PIPNet/cub200 --use_bbox --crop --training_phase joint --explanation_type prp --num_workers 16 --topk_prototype_images"
    
    "--experiment_sub_dir PIPNetNC/cub200 --use_bbox --training_phase joint --explanation_type prp --num_workers 16 --general --compactness --contrastivity --complexity"
    "--experiment_sub_dir PIPNetNC/cub200 --use_bbox --training_phase joint --explanation_type prp --num_workers 16 --topk_prototype_images"
    
    "--experiment_sub_dir PIPNet/dogs --use_bbox --crop --training_phase joint --explanation_type prp --num_workers 16 --general --compactness --contrastivity"
    "--experiment_sub_dir PIPNet/dogs --use_bbox --crop --training_phase joint --explanation_type prp --num_workers 16 --topk_prototype_images"
    
    "--experiment_sub_dir PIPNetNC/dogs --use_bbox --training_phase joint --explanation_type prp --num_workers 16 --general --compactness --contrastivity"
    "--experiment_sub_dir PIPNetNC/dogs --use_bbox --training_phase joint --explanation_type prp --num_workers 16 --topk_prototype_images"
    
    "--experiment_sub_dir PIPNet/cars196 --use_bbox --crop --training_phase joint --explanation_type prp --num_workers 16 --general --compactness --contrastivity"
    "--experiment_sub_dir PIPNet/cars196 --use_bbox --crop --training_phase joint --explanation_type prp --num_workers 16 --topk_prototype_images"
    
    "--experiment_sub_dir PIPNetNC/cars196 --use_bbox --training_phase joint --explanation_type prp --num_workers 16 --general --compactness --contrastivity"
    "--experiment_sub_dir PIPNetNC/cars196 --use_bbox --training_phase joint --explanation_type prp --num_workers 16 --topk_prototype_images"

    # ProtoPool
    "--experiment_sub_dir ProtoPool/cub200 --use_bbox --crop --training_phase fine_tune --explanation_type prp --num_workers 16 --general --compactness --contrastivity --complexity"
    "--experiment_sub_dir ProtoPool/cub200 --use_bbox --crop --training_phase fine_tune --explanation_type prp --num_workers 16 --topk_prototype_images"

    "--experiment_sub_dir ProtoPoolNC/cub200 --use_bbox --training_phase fine_tune --explanation_type prp --num_workers 16 --general --compactness --contrastivity --complexity"
    "--experiment_sub_dir ProtoPoolNC/cub200 --use_bbox --training_phase fine_tune --explanation_type prp --num_workers 16 --topk_prototype_images"

    "--experiment_sub_dir ProtoPool/dogs --use_bbox --crop --training_phase fine_tune --explanation_type prp --num_workers 16 --general --compactness --contrastivity"
    "--experiment_sub_dir ProtoPool/dogs --use_bbox --crop --training_phase fine_tune --explanation_type prp --num_workers 16 --topk_prototype_images"

    "--experiment_sub_dir ProtoPoolNC/dogs --use_bbox --training_phase fine_tune --explanation_type prp --num_workers 16 --general --compactness --contrastivity"
    "--experiment_sub_dir ProtoPoolNC/dogs --use_bbox --training_phase fine_tune --explanation_type prp --num_workers 16 --topk_prototype_images"

    "--experiment_sub_dir ProtoPool/cars196 --use_bbox --crop --training_phase fine_tune --explanation_type prp --num_workers 16 --general --compactness --contrastivity"
    "--experiment_sub_dir ProtoPool/cars196 --use_bbox --crop --training_phase fine_tune --explanation_type prp --num_workers 16 --topk_prototype_images"

    "--experiment_sub_dir ProtoPoolNC/cars196 --use_bbox --training_phase fine_tune --explanation_type prp --num_workers 16 --general --compactness --contrastivity"
    "--experiment_sub_dir ProtoPoolNC/cars196 --use_bbox --training_phase fine_tune --explanation_type prp --num_workers 16 --topk_prototype_images"

    # ProtoPNet
    "--experiment_sub_dir ProtoPNet/cub200 --use_bbox --crop --training_phase fine_tune --explanation_type prp --num_workers 16 --general --compactness --contrastivity --complexity"
    "--experiment_sub_dir ProtoPNet/cub200 --use_bbox --crop --training_phase fine_tune --explanation_type prp --num_workers 16 --topk_prototype_images"

    "--experiment_sub_dir ProtoPNetNC/cub200 --use_bbox --training_phase fine_tune --explanation_type prp --num_workers 16 --general --compactness --contrastivity --complexity"
    "--experiment_sub_dir ProtoPNetNC/cub200 --use_bbox --training_phase fine_tune --explanation_type prp --num_workers 16 --topk_prototype_images"

    "--experiment_sub_dir ProtoPNet/dogs --use_bbox --crop --training_phase fine_tune --explanation_type prp --num_workers 16 --general --compactness --contrastivity"
    "--experiment_sub_dir ProtoPNet/dogs --use_bbox --crop --training_phase fine_tune --explanation_type prp --num_workers 16 --topk_prototype_images"

    "--experiment_sub_dir ProtoPNetNC/dogs --use_bbox --training_phase fine_tune --explanation_type prp --num_workers 16 --general --compactness --contrastivity"
    "--experiment_sub_dir ProtoPNetNC/dogs --use_bbox --training_phase fine_tune --explanation_type prp --num_workers 16 --topk_prototype_images"

    "--experiment_sub_dir ProtoPNet/cars196 --use_bbox --crop --training_phase fine_tune --explanation_type prp --num_workers 16 --general --compactness --contrastivity"
    "--experiment_sub_dir ProtoPNet/cars196 --use_bbox --crop --training_phase fine_tune --explanation_type prp --num_workers 16 --topk_prototype_images"

    "--experiment_sub_dir ProtoPNetNC/cars196 --use_bbox --training_phase fine_tune --explanation_type prp --num_workers 16 --general --compactness --contrastivity"
    "--experiment_sub_dir ProtoPNetNC/cars196 --use_bbox --training_phase fine_tune --explanation_type prp --num_workers 16 --topk_prototype_images"
)

# Iterate over the scripts and run them
for ((i = 0; i < ${#script_arguments[@]}; i++)); do
    arguments="${script_arguments[$i]}"
    echo "Running $script_name with arguments: $arguments"
    python3 "$script_name" $arguments
done