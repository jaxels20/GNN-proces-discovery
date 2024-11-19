#!/bin/bash
#SBATCH --job-name=model_training
#SBATCH --output=/ceph/project/pm-project/result_%j.out
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G

PYTHON_SCRIPT_PATH="training.py"
USER="fkowal20"

singularity exec --nv /ceph/project/pm-project/pm-container-1.tar.sif python3 training.py
