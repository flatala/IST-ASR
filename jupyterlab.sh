#!/bin/bash

#SBATCH --job-name=jupyterlab
#SBATCH --partition=gpu-a100
#SBATCH --time=02:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-task=1
#SBATCH --mem-per-cpu=6G
#SBATCH --account=education-eemcs-courses-dsait4095

# Load modules:
module load 2023r1
module load miniconda3
# module load cuda/11.6
# module load openmpi/4.1.4
# module load py-torch/1.12.1
# module load py-pip
# module load py-numpy
# module load py-pyyaml
# module load py-tqdm
# module load ffmpeg

unset CONDA_SHLVL
source "$(conda info --base)/etc/profile.d/conda.sh"

conda activate IST-ASR
cat /etc/hosts
jupyter lab --ip=0.0.0.0 --port=8888
conda deactivate