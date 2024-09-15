#!/bin/sh
# The following lines instruct Slurm 
#SBATCH --job-name=vanilla
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
split=train
# vanilla oracle report
python3 vanilla_oracle.py \
    --multi_news_file ${DATASET_DIR}/multi_news \
    --split ${split} \
    --output_file outputs/mdrag-5K-${split}-oracle-report.jsonl \
    --quick_test 5000 \
    --tag report 

# vanilla oracle documents
python3 vanilla_oracle.py \
    --multi_news_file ${DATASET_DIR}/multi_news \
    --split ${split} \
    --output_file outputs/mdrag-5K-${split}-oracle-documents.jsonl \
    --quick_test 5000 \
    --tag documents

# vanilla oracle passages
python3 vanilla_oracle.py \
    --multi_news_file ${DATASET_DIR}/multi_news \
    --qrel ${DATASET_DIR}/mdrag-5K/ranking/${split}_qrels_oracle_context_pr.txt \
    --collection ${DATASET_DIR}/mdrag-5K/passages/${split}_psgs.jsonl \
    --split ${split} \
    --quick_test 5000 \
    --output_file outputs/mdrag-5K-${split}-oracle-passages.jsonl \
    --tag passages
