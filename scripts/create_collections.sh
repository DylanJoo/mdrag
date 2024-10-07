#!/bin/sh
# The following lines instruct Slurm 
#SBATCH --job-name=create-collection
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
## Test: Multi-News 
python3 augmentation/create_collections.py \
    --dataset_dir ${DATASET_DIR}/mdrag/shard_data/ratings-gen \
    --split test \
    --output_dir ${DATASET_DIR}/mdrag

## Testb: TREC DUC 04
python3 augmentation/create_collections.py \
    --dataset_dir ${DATASET_DIR}/mdrag/shard_data/ratings-gen \
    --split testb \
    --output_dir ${DATASET_DIR}/mdrag
