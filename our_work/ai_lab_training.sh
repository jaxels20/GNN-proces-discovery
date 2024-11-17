#!/bin/bash
#SBATCH --job-name=model_training
#SBATCH --output=result_%j.out
#SBATCH --error=error_%j.err
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G

PYTHON_SCRIPT="training.py"
VIRTUAL_ENVIRONMENT_NAME="project"
USER="fkowal20"
CONTAINER_NAME="pytorch_24.09.sif"

PROJECT_PATH="/ceph/home/student.aau.dk/$USER/$VIRTUAL_ENVIRONMENT_NAME"
CONTAINER_PATH="/ceph/container/pytorch/$CONTAINER_NAME"
BIND_PATH="$PROJECT_PATH:/project"

singularity exec --bind $BIND_PATH $CONTAINER_PATH $PYTHON $PYTHON_SCRIPT