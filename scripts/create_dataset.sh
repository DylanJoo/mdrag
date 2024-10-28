#!/bin/sh
# The following lines instruct Slurm 
#SBATCH --job-name=create-data
#SBATCH --cpus-per-task=32
#SBATCH --nodes=1
#SBATCH --mem=32G
#SBATCH --ntasks-per-node=1
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x-%j.out

# Set-up the environment.
source ${HOME}/.bashrc
conda activate rag

python3 augmentation/create_collections.py \
    --shard_dir ${DATASET_DIR}/RACE/shard_data \
    --ratings_dir ${DATASET_DIR}/RACE/shard_data/ratings-gen \
    --split test \
    --output_dir ${DATASET_DIR}/RACE
python3 augmentation/create_context_ranking_data.py \
    --shard_dir ${DATASET_DIR}/RACE/shard_data \
    --ratings_dir ${DATASET_DIR}/RACE/shard_data/ratings-gen \
    --split test \
    --output_dir ${DATASET_DIR}/RACE/ranking \
    --n_max_distractors 0 \
    --threshold 3

python3 augmentation/create_collections.py \
    --shard_dir ${DATASET_DIR}/RACE/shard_data \
    --ratings_dir ${DATASET_DIR}/RACE/shard_data/ratings-gen \
    --split testb \
    --output_dir ${DATASET_DIR}/RACE
python3 augmentation/create_context_ranking_data.py \
    --shard_dir ${DATASET_DIR}/RACE/shard_data \
    --ratings_dir ${DATASET_DIR}/RACE/shard_data/ratings-gen \
    --split testb \
    --output_dir ${DATASET_DIR}/RACE/ranking \
    --n_max_distractors 0 \
    --threshold 3
