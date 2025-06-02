#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=128
#SBATCH --gres=gpu:1

cd "$SLURM_SUBMIT_DIR"

python3 prune_protopnet.py
