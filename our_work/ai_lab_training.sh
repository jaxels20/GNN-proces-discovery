#!/bin/bash
#SBATCH --job-name=model_training
#SBATCH --output=/ceph/project/pm-project/result_%j.out
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1
#SBATCH --mem=64G

PYTHON_SCRIPT_PATH="training.py"
USER="fkowal20"

singularity exec /ceph/project/pm-project/pm-container.sif python3 training.py