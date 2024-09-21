#!/bin/sh
# The following lines instruct Slurm 
#SBATCH --job-name=eval-know
#SBATCH --cpus-per-task=16
#SBATCH --nodes=1
#SBATCH --mem=16G
#SBATCH --ntasks-per-node=1
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x-%j.out

# Set-up the environment.
source ${HOME}/.bashrc
conda activate rag
cd ~/mdrag

split=test
# # oracle-report
# python3 eval/judge.py \
#     --judgement_file judgements/mdrag-5K-${split}-oracle-report.jsonl \
#     --dataset_file ${DATASET_DIR}/mdrag-5K/ratings-gen/metallama3.1-8b-${split}.jsonl \
#     --topics ${DATASET_DIR}/mdrag-5K/ranking/${split}_topics_report_request.tsv \
#     --qrels ${DATASET_DIR}/mdrag-5K/ranking/${split}_qrels_oracle_context_pr.txt \
#     --generator_name meta-llama/Meta-Llama-3.1-8B-Instruct \
#     --threshold 3 \
#     --rel_threshold 2 \
#     --tag report
#
# # oracle-passages
# python3 eval/judge.py \
#     --dataset_file ${DATASET_DIR}/mdrag-5K/ratings-gen/metallama3.1-8b-${split}.jsonl \
#     --topics ${DATASET_DIR}/mdrag-5K/ranking/${split}_topics_report_request.tsv \
#     --qrels ${DATASET_DIR}/mdrag-5K/ranking/${split}_qrels_oracle_context_pr.txt \
#     --generator_name meta-llama/Meta-Llama-3.1-8B-Instruct \
#     --threshold 3 \
#     --rel_threshold 2 \
#     --tag passages
#
# # oracle-passages-min
# python3 eval/judge.py \
#     --dataset_file ${DATASET_DIR}/mdrag-5K/ratings-gen/metallama3.1-8b-${split}.jsonl \
#     --topics ${DATASET_DIR}/mdrag-5K/ranking/${split}_topics_report_request.tsv \
#     --qrels ${DATASET_DIR}/mdrag-5K/ranking/${split}_qrels_oracle_context_pr.txt \
#     --generator_name meta-llama/Meta-Llama-3.1-8B-Instruct \
#     --threshold 3 \
#     --rel_threshold 2 \
#     --tag passages-min

# bm25 top10 vanilla
python3 eval/judge.py \
    --judgement_file judgements/mdrag-5K-${split}-bm25-top10-vanilla.jsonl \
    --dataset_file ${DATASET_DIR}/mdrag-5K/ratings-gen/metallama3.1-8b-${split}.jsonl \
    --topics ${DATASET_DIR}/mdrag-5K/ranking/${split}_topics_report_request.tsv \
    --qrels ${DATASET_DIR}/mdrag-5K/ranking/${split}_qrels_oracle_context_pr.txt \
    --generator_name meta-llama/Meta-Llama-3.1-8B-Instruct \
    --threshold 3 \
    --tag bm25-top10-vanilla

# bm25 top10 bartcnndm
python3 eval/judge.py \
    --judgement_file judgements/mdrag-5K-${split}-bm25-top10-bartsum.jsonl \
    --dataset_file ${DATASET_DIR}/mdrag-5K/ratings-gen/metallama3.1-8b-${split}.jsonl \
    --topics ${DATASET_DIR}/mdrag-5K/ranking/${split}_topics_report_request.tsv \
    --qrels ${DATASET_DIR}/mdrag-5K/ranking/${split}_qrels_oracle_context_pr.txt \
    --generator_name meta-llama/Meta-Llama-3.1-8B-Instruct \
    --threshold 3 \
    --tag bm25-top10-bartsum

# bm25 top10 recomp
python3 eval/judge.py \
    --judgement_file judgements/mdrag-5K-${split}-bm25-top10-recomp.jsonl \
    --dataset_file ${DATASET_DIR}/mdrag-5K/ratings-gen/metallama3.1-8b-${split}.jsonl \
    --topics ${DATASET_DIR}/mdrag-5K/ranking/${split}_topics_report_request.tsv \
    --qrels ${DATASET_DIR}/mdrag-5K/ranking/${split}_qrels_oracle_context_pr.txt \
    --generator_name meta-llama/Meta-Llama-3.1-8B-Instruct \
    --threshold 3 \
    --tag bm25-top10-recomp

# contriever top10 bartsum
python3 eval/judge.py \
    --judgement_file judgements/mdrag-5K-${split}-contriever-top10-bartsum.jsonl \
    --dataset_file ${DATASET_DIR}/mdrag-5K/ratings-gen/metallama3.1-8b-${split}.jsonl \
    --topics ${DATASET_DIR}/mdrag-5K/ranking/${split}_topics_report_request.tsv \
    --qrels ${DATASET_DIR}/mdrag-5K/ranking/${split}_qrels_oracle_context_pr.txt \
    --generator_name meta-llama/Meta-Llama-3.1-8B-Instruct \
    --threshold 3 \
    --tag contriever-top10-bartsum

# contriever top10 recomp
python3 eval/judge.py \
    --judgement_file judgements/mdrag-5K-${split}-contriever-top10-recomp.jsonl \
    --dataset_file ${DATASET_DIR}/mdrag-5K/ratings-gen/metallama3.1-8b-${split}.jsonl \
    --topics ${DATASET_DIR}/mdrag-5K/ranking/${split}_topics_report_request.tsv \
    --qrels ${DATASET_DIR}/mdrag-5K/ranking/${split}_qrels_oracle_context_pr.txt \
    --generator_name meta-llama/Meta-Llama-3.1-8B-Instruct \
    --threshold 3 \
    --tag contriever-top10-recomp
