#!/bin/bash
#SBATCH --job-name=model_training
#SBATCH --output=/ceph/project/pm-project/training_output_%j.out
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=15
#SBATCH --mem=24G

singularity exec --nv /ceph/project/pm-project/pm-container-1.tar.sif python3 training.py
