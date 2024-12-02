#!/bin/bash -l
#SBATCH --job-name=long              # Job name
#SBATCH --output=testing/o%A_%a     # Name of stdout output file
#SBATCH --error=testing/e%A_%a      # Name of stderr error file
#SBATCH --partition=standard-g         # partition name
#SBATCH --nodes=1                   # Total number of nodes 
#SBATCH --ntasks-per-node=8         # 8 MPI ranks per node, 16 total (2x8)
#SBATCH --array=1-1%1
#SBATCH --gpus-per-node=8           # Allocate one gpu per MPI rank
#SBATCH --time=3-00:00:00           # Run time (d-hh:mm:ss)
#SBATCH --account=project_465001396 # Project for billing

source ${HOME}/.bashrc
cd ~/mdrag
set HF_TOKEN=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

SIF=/scratch/project_465001396/images/rocm-vllm_ubuntu22.04_rocm6.2_py3.10_torch2.3.0_vllm0.5.5.sif
DATASET_DIR=/project/project_465001396/dylan
HPARAMS_FILE=${HOME}/hparams.txt

# Start the experiment.
singularity exec --bind /scratch/project_465001396,/project/project_465001396 ${SIF} \
    python augmentation/gen_passages.py \
        --shard 0 --shard_size 30 \
        --config configs/mds-decontextualize.llama3-8b.yaml \
        --multi_news_file /scratch/project_465001396/dylan/datasets/multi_news \
        --split train \
        --model meta-llama/Meta-Llama-3.1-70B-Instruct \
        --model_tag metallama3.1-8b \
        --tag psgs-gen \
        --load_mode vllm \
        --temperature 0.7 \
        --max_new_tokens 640 \
        --output_dir ${DATASET_DIR}/mdrag/shard_data/ \
        --num_gpus 8 \
        --ampere_gpu
