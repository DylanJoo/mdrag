#!/bin/sh
# The following lines instruct Slurm 
#SBATCH --job-name=eval
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

# [Bartsum]
for aug_method in bartsum;do
for retriever in contriever;do
for topk in 10;do
    split=testb
    python3 -m evaluation.llm_prejudge \
        --generator_name meta-llama/Meta-Llama-3.1-8B-Instruct \
        --run_file runs/baseline.${retriever}.race-${split}.passages.run \
        --topk ${topk} \
        --weighted_factor 0.25 \
        --split ${split} \
        --judgement_file judgements/${split}_${aug_method}_judgements.jsonl \
        --threshold 3 \
        --qrels ${DATASET_DIR}/RACE/ranking/${split}_qrels_oracle_context_pr.txt \
        --rel_threshold 3 \
        --passage_path outputs/${split}_${aug_method}_psgs.jsonl \
        --dataset_dir ${DATASET_DIR}/RACE \
        --tag ${retriever}-${topk}-${aug_method} 
done
done
done
