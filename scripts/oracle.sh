#!/bin/sh
# The following lines instruct Slurm 
#SBATCH --job-name=oracle
#SBATCH --cpus-per-task=32
#SBATCH --nodes=1
#SBATCH --mem=32G
#SBATCH --ntasks-per-node=1
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x-%j.out

# Set-up the environment.
source ${HOME}/.bashrc
conda activate rag
cd ~/mdrag

# Start the experiment.
split=test
python3 oracle.py \
    --multi_news_file ${DATASET_DIR}/multi_news \
    --split ${split} \
    --output_file outputs/race-${split}-oracle-report.jsonl \
    --tag report 

python3 oracle.py \
    --multi_news_file ${DATASET_DIR}/multi_news \
    --split ${split} \
    --output_file outputs/race-${split}-oracle-documents.jsonl \
    --tag documents

python3 oracle.py \
    --multi_news_file ${DATASET_DIR}/multi_news \
    --qrel ${DATASET_DIR}/RACE/ranking/${split}_qrels_oracle_context_pr.txt \
    --collection ${DATASET_DIR}/RACE/passages \
    --split ${split} \
    --output_file outputs/race-${split}-oracle-passages.jsonl \
    --tag passages

split=testb
python3 oracle.py \
    --duc04_file ${DATASET_DIR}/duc04 \
    --split ${split} \
    --output_file outputs/race-${split}-oracle-report.jsonl \
    --tag report 

python3 oracle.py \
    --duc04_file ${DATASET_DIR}/duc04 \
    --split ${split} \
    --output_file outputs/race-${split}-oracle-documents.jsonl \
    --tag documents

python3 oracle.py \
    --duc04_file ${DATASET_DIR}/duc04 \
    --qrel ${DATASET_DIR}/RACE/ranking/${split}_qrels_oracle_context_pr.txt \
    --collection ${DATASET_DIR}/RACE/passages \
    --split ${split} \
    --output_file outputs/race-${split}-oracle-passages.jsonl \
    --tag passages
