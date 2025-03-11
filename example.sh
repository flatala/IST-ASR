#!/bin/bash

#SBATCH --job-name=example
#SBATCH --partition=gpu
#SBATCH --time=00:15:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-task=1
#SBATCH --mem-per-cpu=1G
#SBATCH --account=education-eemcs-courses-dsait4095

# Load modules:
module load 2023r1
module load cuda/11.6
module load openmpi/4.1.4
module load py-torch/1.12.1
module load py-pip
module load py-numpy
module load py-pandas
module load py-pyyaml
module load py-tqdm
module load ffmpeg

python TORGO_prepare.py 'TORGO 
