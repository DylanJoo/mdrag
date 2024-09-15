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
for split in train test;do
    python3 augmentation/create_collections.py \
        --dataset_file ${DATASET_DIR}/mdrag-5K/ratings-gen/metallama3.1-8b-${split}.jsonl \
        --output_dir ${DATASET_DIR}/mdrag-5K
done
