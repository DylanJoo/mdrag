#!/bin/bash -l
#SBATCH --job-name=topics              # Job name
#SBATCH --output=train-logs-t/o%A_%a     # Name of stdout output file
#SBATCH --error=train-logs-t/e%A_%a      # Name of stderr error file
#SBATCH --partition=small-g         # partition name
#SBATCH --nodes=1                   # Total number of nodes 
#SBATCH --ntasks-per-node=1         # 8 MPI ranks per node, 16 total (2x8)
#SBATCH --array=1-40%8
#SBATCH --gpus-per-node=1           # Allocate one gpu per MPI rank
#SBATCH --time=3-00:00:00           # Run time (d-hh:mm:ss)
#SBATCH --account=project_465001339 # Project for billing

source ${HOME}/.bashrc
cd ~/mdrag

SIF=~/temp/rocm-vllm_ubuntu22.04_rocm6.2_py3.10_torch2.3.0_vllm0.5.5.sif
DATASET_DIR=/project/project_465001339/datasets
HPARAMS_FILE=${HOME}/hparams_full.txt

# Start the experiment.
singularity exec --bind /scratch/project_465001339,/project/project_465001339 ${SIF} \
    python augmentation/gen_topics.py \
        $(head -$SLURM_ARRAY_TASK_ID $HPARAMS_FILE | tail -1) \
        --config configs/mds-decontextualize.llama3-8b.yaml \
        --split train \
        --model meta-llama/Meta-Llama-3.1-8B-Instruct \
        --model_tag metallama3.1-8b \
        --tag topics-gen \
        --load_mode vllm \
        --temperature 0.7 \
        --max_new_tokens 128 \
        --output_dir ${DATASET_DIR}/mdrag/shard_data/ \
        --num_gpus 1 \
        --ampere_gpu
