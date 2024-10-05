#!/bin/sh
# The following lines instruct Slurm to allocate one GPU.
#SBATCH --job-name=rating-gen
#SBATCH --partition gpu
#SBATCH --gres=gpu:nvidia_l40:1
#SBATCH --mem=32G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=120:00:00
#SBATCH --output=logs/%x-%j.out

# Set-up the environment.
source ${HOME}/.bashrc
conda activate rag
cd ~/mdrag

for split in train;do
    python3 augmentation/gen_ratings.py \
        --shard_dir ${DATASET_DIR}/mdrag/shard_data \
        --config configs/mds-decontextualize.llama3-8b-chat.yaml \
        --split ${split} \
        --model meta-llama/Meta-Llama-3.1-8B-Instruct \
        --model_tag metallama3.1-8b \
        --tag ratings-gen \
        --load_mode vllm \
        --temperature 0.7 \
        --top_p 0.9 \
        --max_new_tokens 5 \
        --output_dir ${DATASET_DIR}/mdrag/ \
        --ampere_gpu
done
