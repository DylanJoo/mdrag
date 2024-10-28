#!/bin/sh
# The following lines instruct Slurm to allocate one GPU.
#SBATCH --job-name=oracle
#SBATCH --partition gpu
#SBATCH --gres=gpu:nvidia_rtx_a6000:1
#SBATCH --mem=32G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=120:00:00
#SBATCH --output=logs/%x.%j.out

# Set-up the environment.
source ${HOME}/.bashrc
conda activate rag
cd ~/mdrag

## 1. Extract the oracle context (report)
# split=test
# python3 oracle.py \
#     --multi_news_file ${DATASET_DIR}/multi_news \
#     --split ${split} \
#     --output_file outputs/race-${split}-oracle-report.jsonl \
#     --tag report 
#
# split=testb
# python3 oracle.py \
#     --duc04_file ${DATASET_DIR}/duc04 \
#     --split ${split} \
#     --output_file outputs/race-${split}-oracle-report.jsonl \
#     --tag report 

## 2. Judge the context using llama3.1-8b (GPU)
## [Note] testb has 15 EXAM questions
# split=test
# context_file=outputs/race-${split}-oracle-report.jsonl
# python3 augmentation/judge_ratings.py \
#     --shard_dir ${DATASET_DIR}/RACE/shard_data \
#     --config configs/mds-decontextualize.llama3-8b.yaml \
#     --context_file ${context_file} \
#     --topics ${DATASET_DIR}/RACE/ranking_1/${split}_topics_report_request.tsv \
#     --output_file ${context_file/outputs/judgements} \
#     --split ${split} \
#     --model meta-llama/Meta-Llama-3.1-8B-Instruct \
#     --model_tag metallama3.1-8b \
#     --load_mode vllm \
#     --temperature 0.7 \
#     --top_p 0.9 \
#     --max_new_tokens 5 \
#     --ampere_gpu

# split=testb
# context_file=outputs/race-${split}-oracle-report.jsonl
# python3 augmentation/judge_ratings.py \
#     --shard_dir ${DATASET_DIR}/RACE/shard_data \
#     --config configs/mds-decontextualize.llama3-8b.yaml \
#     --context_file ${context_file} \
#     --topics ${DATASET_DIR}/RACE/ranking_1/${split}_topics_report_request.tsv \
#     --output_file ${context_file/outputs/judgements} \
#     --n_questions 15 \
#     --split ${split} \
#     --model meta-llama/Meta-Llama-3.1-8B-Instruct \
#     --model_tag metallama3.1-8b \
#     --load_mode vllm \
#     --temperature 0.7 \
#     --top_p 0.9 \
#     --max_new_tokens 5 \
#     --ampere_gpu

## 3. Evaluate the RACE knowledge
# oracle-report
for split in test testb;do
python3 -m evaluation \
    --judgement_file judgements/race-${split}-oracle-report.jsonl \
    --dataset_file ${DATASET_DIR}/RACE/shard_data/ratings-gen/8b/metallama3.1-8b-${split}-0.jsonl \
    --topics ${DATASET_DIR}/RACE/ranking_1/${split}_topics_report_request.tsv \
    --qrels ${DATASET_DIR}/RACE/ranking/${split}_qrels_oracle_context_pr.txt \
    --generator_name meta-llama/Meta-Llama-3.1-8B-Instruct \
    --threshold 3 \
    --tag report

# oracle-passages
python3 -m evaluation \
    --dataset_file ${DATASET_DIR}/RACE/shard_data/ratings-gen/8b/metallama3.1-8b-${split}-0.jsonl \
    --topics ${DATASET_DIR}/RACE/ranking_1/${split}_topics_report_request.tsv \
    --qrels ${DATASET_DIR}/RACE/ranking/${split}_qrels_oracle_context_pr.txt \
    --generator_name meta-llama/Meta-Llama-3.1-8B-Instruct \
    --threshold 3 \
    --tag passages

# oracle-passages-min
python3 -m evaluation \
    --dataset_file ${DATASET_DIR}/RACE/shard_data/ratings-gen/8b/metallama3.1-8b-${split}-0.jsonl \
    --topics ${DATASET_DIR}/RACE/ranking_1/${split}_topics_report_request.tsv \
    --qrels ${DATASET_DIR}/RACE/ranking/${split}_qrels_oracle_context_pr.txt \
    --generator_name meta-llama/Meta-Llama-3.1-8B-Instruct \
    --threshold 3 \
    --rel_threshold 3 \
    --tag passages-min
done
