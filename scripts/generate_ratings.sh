#!/bin/bash -l
#SBATCH --job-name=ratings            # Job name
#SBATCH --output=test-logs/ratings.o%j   # Name of stdout output file
#SBATCH --error=test-logs/ratings.e%j    # Name of stderr error file
#SBATCH --partition=small-g         # partition name
#SBATCH --nodes=1                   # Total number of nodes 
#SBATCH --ntasks-per-node=1         # 8 MPI ranks per node, 16 total (2x8)
#SBATCH --gpus-per-node=1           # Allocate one gpu per MPI rank
#SBATCH --time=3-00:00:00           # Run time (d-hh:mm:ss)
#SBATCH --account=project_465001339 # Project for billing

source ${HOME}/.bashrc
cd ~/mdrag

SIF=~/temp/rocm-vllm_ubuntu22.04_rocm6.2_py3.10_torch2.3.0_vllm0.5.5.sif
DATASET_DIR=/project/project_465001339/datasets
HPARAMS_FILE=${HOME}/hparams.txt

# Start the experiment.
singularity exec --bind /scratch/project_465001339,/project/project_465001339 ${SIF} \
    python3 augmentation/gen_ratings.py \
        --shard_dir ${DATASET_DIR}/mdrag/shard_data \
        --config configs/mds-decontextualize.llama3-8b.yaml \
        --split testb \
        --model meta-llama/Meta-Llama-3.1-8B-Instruct \
        --model_tag metallama3.1-8b \
        --tag ratings-gen \
        --load_mode vllm \
        --temperature 0.7 \
        --max_new_tokens 3 \
        --output_dir ${DATASET_DIR}/mdrag/ \
        --num_gpus 1 \
        --n_questions 15 \
        --ampere_gpu

singularity exec --bind /scratch/project_465001339,/project/project_465001339 ${SIF} \
    python3 augmentation/gen_ratings.py \
        --shard_dir ${DATASET_DIR}/mdrag/shard_data \
        --config configs/mds-decontextualize.llama3-8b.yaml \
        --split test \
        --model meta-llama/Meta-Llama-3.1-8B-Instruct \
        --model_tag metallama3.1-8b \
        --tag ratings-gen \
        --load_mode vllm \
        --temperature 0.7 \
        --max_new_tokens 3 \
        --output_dir ${DATASET_DIR}/mdrag/ \
        --num_gpus 1 \
        --ampere_gpu
