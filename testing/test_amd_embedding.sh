<<<<<<< Updated upstream
#!/bin/sh
# The following lines instruct Slurm to allocate one GPU.
#SBATCH --job-name=testing-e
#SBATCH --partition gpu
#SBATCH --gres=gpu:nvidia_rtx_a6000:1
#SBATCH --mem=32G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=120:00:00
#SBATCH --output=logs/%x.%j.out
=======
#!/bin/bash -l
#SBATCH --job-name=examplejob        # Job name
#SBATCH --output=examplejob.o%j      # Name of stdout output file
#SBATCH --error=examplejob.e%j       # Name of stderr error file
#SBATCH --partition=standard-g       # partition name
#SBATCH --nodes=1                    # Total number of nodes 
#SBATCH --ntasks-per-node=8          # 8 MPI ranks per node, 16 total (2x8)
#SBATCH --gpus-per-node=8            # Allocate one gpu per MPI rank
#SBATCH --time=1-12:00:00            # Run time (d-hh:mm:ss)
#SBATCH --account=project_465001339  # Project for billing
>>>>>>> Stashed changes

# Set-up the environment.
source ${HOME}/.bashrc
cd ~/mdrag/testing
<<<<<<< Updated upstream
conda activate rag

python test_amd_embedding.py
=======
conda activate vllm

# export PYTORCH_ROCM_ARCH="gfx90a;gfx942"
# export PATH=$PATH:/opt/rocm-6.0.3/bin

# SIF=~/temp/vllm_rocm6.2_mi300_ubuntu22.04_py3.9_vllm_7c5fd50.sif
# SIF=~/temp/pytorch_rocm6.1.2_ubuntu20.04_py3.9_pytorch_staging.sif
SIF=~/temp/vllm-rocm_vllm-v0.2.6.sif
# SIF=~/temp/rocm6.1.2_py3.10_torch2.5_vllm0.5_bkc_v2.0.sif

singularity exec --bind /scratch/project_465001339 ${SIF} python test_amd_embedding.py
# singularity exec --bind /scratch/project_465001339 ${SIF} \
#     pip freeze > requirement0.txt
>>>>>>> Stashed changes
