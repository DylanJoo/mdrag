#!/bin/sh
# The following lines instruct Slurm 
#SBATCH --job-name=create-context-rank
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

# flatten generated passages
python3 augmentation/create_context_ranking_data.py \
    --shard_dir ${DATASET_DIR}/mdrag-5K/shard_data \
    --dataset_file ${DATASET_DIR}/mdrag-5K/ratings-gen/metallama3.1-8b-train.jsonl \
    --output_dir ${DATASET_DIR}/mdrag-5K/ranking \
    --n_max_distractors 5 \
    --threshold 3 \
    --doc_lucene_index ${INDEX_DIR}/mdrag-5K-documents.lucene

