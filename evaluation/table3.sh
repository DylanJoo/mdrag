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

rm test.table3
rm testb.table3

aug_method=vanilla+reranking
for topk in 10;do
for retriever in bm25 contriever splade;do
    split=test
    python3 -m evaluation.llm_prejudge \
        --generator_name meta-llama/Meta-Llama-3.1-8B-Instruct \
        --judgement_file ${DATASET_DIR}/RACE/ranking/${split}_judgements.jsonl \
        --threshold 3 \
        --run_file runs/reranking.${retriever}+monoT5.race-${split}.passages.run \
        --topk ${topk} \
        --qrels ${DATASET_DIR}/RACE/ranking/${split}_qrels_oracle_context_pr.txt \
        --n_questions 10 \
        --passage_path ${DATASET_DIR}/RACE/passages \
        --tag ${retriever}-${topk}-${aug_method} >> ${split}.table3
    split=testb
    python3 -m evaluation.llm_prejudge \
        --generator_name meta-llama/Meta-Llama-3.1-8B-Instruct \
        --judgement_file ${DATASET_DIR}/RACE/ranking/${split}_judgements.jsonl \
        --threshold 3 \
        --run_file runs/reranking.${retriever}+monoT5.race-${split}.passages.run \
        --topk ${topk} \
        --qrels ${DATASET_DIR}/RACE/ranking/${split}_qrels_oracle_context_pr.txt \
        --n_questions 15 \
        --passage_path ${DATASET_DIR}/RACE/passages \
        --tag ${retriever}-${topk}-${aug_method} >> ${split}.table3
done
done

