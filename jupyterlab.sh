#!/bin/bash

#SBATCH --job-name=jupyterlab
#SBATCH --partition=gpu-a100-small
#SBATCH --time=03:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-task=1
#SBATCH --mem-per-cpu=4G
#SBATCH --account=education-eemcs-courses-dsait4095

# Load modules:
module load 2023r1
module load miniconda3
module load ffmpeg
module load cuda/12.1

unset CONDA_SHLVL
source "$(conda info --base)/etc/profile.d/conda.sh"

conda activate nemo
cat /etc/hosts
jupyter lab --ip=0.0.0.0 --port=8888
conda deactivate