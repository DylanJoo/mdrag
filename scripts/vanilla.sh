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

# Start the experiment.
for split in test;do
    python3 vanilla.py \
        --topics ${DATASET_DIR}/mdrag-5K/ranking/${split}_topics_report_request.tsv \
        --collection ${DATASET_DIR}/mdrag-5K/passages \
        --run retrieval/baseline.bm25.mdrag-5K-${split}.passages.run \
        --topk 10 \
        --output_file outputs/mdrag-5K-${split}-bm25-top10-vanilla.jsonl
done

