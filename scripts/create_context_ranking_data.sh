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

## Multi-News Train and test
for split in test testb;do
    python3 augmentation/create_context_ranking_data.py \
        --shard_dir ${DATASET_DIR}/RACE/shard_data \
        --dataset_dir ${DATASET_DIR}/RACE/shard_data/ratings-gen/8b \
        --split ${split} \
        --output_dir ${DATASET_DIR}/RACE/ranking \
        --n_max_distractors 0 \
        --threshold 3

    ## [ablation] larger grading model
    python3 augmentation/create_context_ranking_data.py \
        --shard_dir ${DATASET_DIR}/RACE/shard_data \
        --dataset_dir ${DATASET_DIR}/RACE/shard_data/ratings-gen \
        --split ${split} \
        --output_dir ${DATASET_DIR}/RACE/ranking/70b \
        --n_max_distractors 0 \
        --threshold 3

    ## [ablation] other threshold
    for threshold in 1 5;do
        python3 augmentation/create_context_ranking_data.py \
            --shard_dir ${DATASET_DIR}/RACE/shard_data \
            --dataset_dir ${DATASET_DIR}/RACE/shard_data/ratings-gen/8b \
            --split ${split} \
            --output_dir ${DATASET_DIR}/RACE/ranking/${threshold} \
            --n_max_distractors 0 \
            --threshold ${threshold} 
    done
done

# split=train
# python3 augmentation/create_context_ranking_data.py \
#     --shard_dir ${DATASET_DIR}/RACE/shard_data \
#     --dataset_dir ${DATASET_DIR}/RACE/shard_data/ratings-gen \
#     --split ${split} \
#     --output_dir ${DATASET_DIR}/RACE/ranking \
#     --n_max_distractors 5 \
#     --threshold 3 \
#     --doc_lucene_index ${INDEX_DIR}/race-passages.lucene

## [NOTE] testb from DUC'04 --> still need to reconsider the collection
