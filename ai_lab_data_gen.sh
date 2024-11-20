#!/bin/bash
#SBATCH --job-name=data_generation
#SBATCH --output=/ceph/project/pm-project/data_gen_output_%j.out
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=15
#SBATCH --mem=64G

singularity exec --nv /ceph/project/pm-project/pm-container-1.tar.sif python3 data_generation/data_generation.py