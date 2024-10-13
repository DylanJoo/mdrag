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

# bm25/contriever/splade top10  + vanilla/bartcnndm/recomp
# for split in test testb;do
for split in testb;do
for retriever in bm25 contriever splade;do
for aug_method in vanilla bartsum recomp;do
    python3 -m evalulation \
        --judgement_file judgements/RACE-${split}-${retriever}-top10-${aug_method}.jsonl \
        --dataset_file ${DATASET_DIR}/RACE/ratings-gen/metallama3.1-8b-${split}.jsonl \
        --topics ${DATASET_DIR}/RACE/ranking/${split}_topics_report_request.tsv \
        --qrels ${DATASET_DIR}/RACE/ranking/${split}_qrels_oracle_context_pr.txt \
        --generator_name meta-llama/Meta-Llama-3.1-8B-Instruct \
        --threshold 3 \
        --tag ${retriever}-top10-${aug_method}
done
done
done
