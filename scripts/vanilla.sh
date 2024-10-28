#!/bin/sh
# The following lines instruct Slurm 
#SBATCH --job-name=vanilla
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
for retriever in bm25 splade contriever;do
    split=testb
    python3 vanilla.py \
        --topics ${DATASET_DIR}/RACE/ranking_1/${split}_topics_report_request.tsv \
        --collection ${DATASET_DIR}/RACE/passages/${split} \
        --run runs/baseline.${retriever}.race-${split}.passages.run \
        --topk 30 \
        --output_file outputs/race-${split}-${retriever}-top30-vanilla.jsonl

    split=test
    python3 vanilla.py \
        --topics ${DATASET_DIR}/RACE/ranking_1/${split}_topics_report_request.tsv \
        --collection ${DATASET_DIR}/RACE/passages/${split} \
        --run runs/baseline.${retriever}.race-${split}.passages.run \
        --topk 30 \
        --output_file outputs/race-${split}-${retriever}-top30-vanilla.jsonl
done

for retriever in bm25 splade contriever;do
    split=testb
    python3 vanilla.py \
        --topics ${DATASET_DIR}/RACE/ranking_1/${split}_topics_report_request.tsv \
        --collection ${DATASET_DIR}/RACE/passages/${split} \
        --run runs/reranking.${retriever}+monoT5.race-${split}.passages.run \
        --topk 30 \
        --output_file outputs/race-${split}-${retriever}+monoT5-top30-vanilla.jsonl

    split=test
    # python3 vanilla.py \
    #     --topics ${DATASET_DIR}/RACE/ranking_1/${split}_topics_report_request.tsv \
    #     --collection ${DATASET_DIR}/RACE/passages/${split} \
    #     --run runs/baseline.${retriever}.race-${split}.passages.run \
    #     --topk 30 \
    #     --output_file outputs/race-${split}-${retriever}-top30-vanilla.jsonl
done

