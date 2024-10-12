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
# oracle-report
python3 evaluation/judge.py \
    --judgement_file judgements/race-${split}-oracle-report.jsonl \
    --dataset_file ${DATASET_DIR}/RACE/ratings-gen/8b/metallama3.1-8b-${split}.jsonl \
    --topics ${DATASET_DIR}/RACE/ranking/${split}_topics_report_request.tsv \
    --qrels ${DATASET_DIR}/RACE/ranking/${split}_qrels_oracle_context_pr.txt \
    --generator_name meta-llama/Meta-Llama-3.1-8B-Instruct \
    --threshold 3 \
    --rel_threshold 2 \
    --tag report

# oracle-passages
python3 evaluation/judge.py \
    --dataset_file ${DATASET_DIR}/RACE/ratings-gen/8b/metallama3.1-8b-${split}.jsonl \
    --topics ${DATASET_DIR}/RACE/ranking/${split}_topics_report_request.tsv \
    --qrels ${DATASET_DIR}/RACE/ranking/${split}_qrels_oracle_context_pr.txt \
    --generator_name meta-llama/Meta-Llama-3.1-8B-Instruct \
    --threshold 3 \
    --rel_threshold 2 \
    --tag passages

# oracle-passages-min
python3 evaluation/judge.py \
    --dataset_file ${DATASET_DIR}/RACE/ratings-gen/8b/metallama3.1-8b-${split}.jsonl \
    --topics ${DATASET_DIR}/RACE/ranking/${split}_topics_report_request.tsv \
    --qrels ${DATASET_DIR}/RACE/ranking/${split}_qrels_oracle_context_pr.txt \
    --generator_name meta-llama/Meta-Llama-3.1-8B-Instruct \
    --threshold 3 \
    --rel_threshold 2 \
    --tag passages-min

# bm25 top10 vanilla/bartcnndm/recomp
# for aug_method in vanilla bartsum recomp;do
#     python3 eval/judge.py \
#         --judgement_file judgements/RACE-${split}-bm25-top10-${aug_method}.jsonl \
#         --dataset_file ${DATASET_DIR}/RACE/ratings-gen/metallama3.1-8b-${split}.jsonl \
#         --topics ${DATASET_DIR}/RACE/ranking/${split}_topics_report_request.tsv \
#         --qrels ${DATASET_DIR}/RACE/ranking/${split}_qrels_oracle_context_pr.txt \
#         --generator_name meta-llama/Meta-Llama-3.1-8B-Instruct \
#         --threshold 3 \
#         --tag bm25-top10-${aug_method}
# done

# contriever top10 vanilla
# for aug_method in vanilla bartsum recomp;do
#     python3 eval/judge.py \
#         --judgement_file judgements/RACE-${split}-contriever-top10-${aug_method}.jsonl \
#         --dataset_file ${DATASET_DIR}/RACE/ratings-gen/metallama3.1-8b-${split}.jsonl \
#         --topics ${DATASET_DIR}/RACE/ranking/${split}_topics_report_request.tsv \
#         --qrels ${DATASET_DIR}/RACE/ranking/${split}_qrels_oracle_context_pr.txt \
#         --generator_name meta-llama/Meta-Llama-3.1-8B-Instruct \
#         --threshold 3 \
#         --tag contriever-top10-${aug_method}
# done

