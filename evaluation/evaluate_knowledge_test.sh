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
aug_method=bartsum
aug_method=recomp
# for retriever in bm25 contriever splade;do
for retriever in bm25;do
for topk in 30;do
    split=testb
    python3 -m evaluation.llm_prejudge \
        --generator_name meta-llama/Meta-Llama-3.1-8B-Instruct \
        --judgement_file judgements/${split}_${aug_method}_judgements.jsonl \
        --threshold 3 \
        --run_file runs/baseline.${retriever}.race-${split}.passages.run \
        --topk ${topk} \
        --qrels ${DATASET_DIR}/RACE/ranking/${split}_qrels_oracle_context_pr.txt \
        --n_questions 15 \
        --passage_path outputs/${split}_${aug_method}_psgs.jsonl \
        --tag ${retriever}-${topk}-${aug_method} 
done
done

