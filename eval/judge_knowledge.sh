#!/bin/sh
# The following lines instruct Slurm to allocate one GPU.
#SBATCH --job-name=rating-judge
#SBATCH --partition gpu
#SBATCH --gres=gpu:nvidia_rtx_a6000:1
#SBATCH --mem=32G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=120:00:00
#SBATCH --output=logs/%x-%j.out

# Set-up the environment.
source ${HOME}/.bashrc
conda activate rag
cd ~/mdrag

split=test
augmentation=vanilla

# rating
for context_file in outputs/mdrag-5K-${split}-bm25-top10-${augmentation}.jsonl; do
    python3 augmentation/judge_ratings.py \
        --shard_dir ${DATASET_DIR}/mdrag-5K/shard_data \
        --config configs/mds-decontextualize.llama3-8b-chat.yaml \
        --context_file ${context_file} \
        --topics ${DATASET_DIR}/mdrag-5K/ranking/${split}_topics_report_request.tsv \
        --output_file ${context_file/outputs/judgements} \
        --split ${split} \
        --model meta-llama/Meta-Llama-3.1-8B-Instruct \
        --model_tag metallama3.1-8b \
        --load_mode vllm \
        --temperature 0.7 \
        --top_p 0.9 \
        --max_new_tokens 5 \
        --ampere_gpu
done
