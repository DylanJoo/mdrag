#!/bin/bash -l
#SBATCH --job-name=examplejob        # Job name
#SBATCH --output=examplejob.o%j      # Name of stdout output file
#SBATCH --error=examplejob.e%j       # Name of stderr error file
#SBATCH --partition=standard-g       # partition name
#SBATCH --nodes=1                    # Total number of nodes 
#SBATCH --ntasks-per-node=4          # 8 MPI ranks per node, 16 total (2x8)
#SBATCH --gpus-per-node=4            # Allocate one gpu per MPI rank
#SBATCH --time=1-12:00:00            # Run time (d-hh:mm:ss)
#SBATCH --account=project_465001339  # Project for billing

# Set-up the environment.
source ${HOME}/.bashrc
cd ~/mdrag/testing
conda activate base
# SIF=~/temp/rocm-vllm_ubuntu20.04_rocm6.1.2_py3.9_vllm0.5.5.sif
# SIF=~/temp/rocm-vllm_ubuntu22.04_rocm6.2_py3.10_torch2.3.0_vllm0.5.5.sif
SIF=~/temp/rocm-vllm_ubuntu22.04_rocm6.2_py3.10_torch2.5.0_09-07-2024_vllm0.5.5.sif

singularity exec --bind /scratch/project_465001339 ${SIF} python test_amd_generation.py
