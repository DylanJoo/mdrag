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

# Compared methods
retriever=*
aug_method=*
for context_file in outputs/race-${split}-${retriever}-top10-${aug_method}.jsonl; do
    python3 augmentation/judge_ratings.py \
        --shard_dir ${DATASET_DIR}/RACE/shard_data \
        --config configs/mds-decontextualize.llama3-8b.yaml \
        --context_file ${context_file} \
        --topics ${DATASET_DIR}/RACE/ranking/${split}_topics_report_request.tsv \
        --output_file ${context_file/outputs/judgements} \
        --n_questions 10 \
        --split ${split} \
        --model meta-llama/Meta-Llama-3.1-8B-Instruct \
        --model_tag metallama3.1-8b \
        --load_mode vllm \
        --temperature 0.7 \
        --top_p 0.9 \
        --max_new_tokens 5 \
        --ampere_gpu
done

split=testb
retriever=*
aug_method=*
for context_file in outputs/race-${split}-${retriever}-top10-${aug_method}.jsonl; do
    python3 augmentation/judge_ratings.py \
        --shard_dir ${DATASET_DIR}/RACE/shard_data \
        --config configs/mds-decontextualize.llama3-8b.yaml \
        --context_file ${context_file} \
        --topics ${DATASET_DIR}/RACE/ranking/${split}_topics_report_request.tsv \
        --output_file ${context_file/outputs/judgements} \
        --n_questions 15 \
        --split ${split} \
        --model meta-llama/Meta-Llama-3.1-8B-Instruct \
        --model_tag metallama3.1-8b \
        --load_mode vllm \
        --temperature 0.7 \
        --top_p 0.9 \
        --max_new_tokens 5 \
        --ampere_gpu
done
