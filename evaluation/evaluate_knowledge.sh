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

# split=test
# aug_method=vanilla
# for k in 10 20 30;do
# for retriever in bm25 contriever splade;do
#     python3 -m evaluation.llm_prejudge \
#         --generator_name meta-llama/Meta-Llama-3.1-8B-Instruct \
#         --run_file runs/baseline.${retriever}.race-${split}.passages.run \
#         --topk 1 \
#         --judgement_file ${DATASET_DIR}/RACE/ranking/${split}_judgements.jsonl \
#         --threshold 3 \
#         --qrels ${DATASET_DIR}/RACE/ranking/${split}_qrels_oracle_context_pr.txt \
#         --n_questions 10 \
#         --passage_dir ${DATASET_DIR}/RACE/passages \
#         --tag ${retriever}-topk-${aug_method}
# done
# done

# [ORACLE]
# split=testb
# for aug_method in oracle-passages;do
#     python3 -m evaluation.llm_prejudge \
#         --generator_name meta-llama/Meta-Llama-3.1-8B-Instruct \
#         --judgement_file ${DATASET_DIR}/RACE/ranking/${split}_judgements.jsonl \
#         --threshold 3 \
#         --qrels ${DATASET_DIR}/RACE/ranking/${split}_qrels_oracle_context_pr.txt \
#         --rel_threshold 3 \
#         --n_questions 15 \
#         --passage_dir ${DATASET_DIR}/RACE/passages \
#         --tag ${aug_method}
# done


# [BARTSUM]
split=testb
for aug_method in oracle-passages;do
    python3 -m evaluation.llm_prejudge \
        --generator_name meta-llama/Meta-Llama-3.1-8B-Instruct \
        --judgement_file ${DATASET_DIR}/RACE/ranking/${split}_judgements.jsonl \
        --threshold 3 \
        --qrels ${DATASET_DIR}/RACE/ranking/${split}_qrels_oracle_context_pr.txt \
        --rel_threshold 3 \
        --n_questions 15 \
        --passage_dir ${DATASET_DIR}/RACE/passages \
        --tag ${aug_method}
done
