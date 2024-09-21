#!/bin/sh
# The following lines instruct Slurm to allocate one GPU.
#SBATCH --job-name=testing-g
#SBATCH --partition gpu
#SBATCH --gres=gpu:nvidia_rtx_a6000:1
#SBATCH --mem=32G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=120:00:00
#SBATCH --output=logs/%x.%j.out

# Set-up the environment.
source ${HOME}/.bashrc
cd ~/mdrag/testing
conda activate rag

python test_amd_generation.py
