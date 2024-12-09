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

# [Oracle]
# aug_method=oracle-report
# split=test
# python3 -m evaluation.llm_prejudge \
#     --generator_name meta-llama/Meta-Llama-3.1-8B-Instruct \
#     --judgement_file judgements/${split}_oracle-report_judgements.jsonl \
#     --threshold 3 \
#     --qrels ${DATASET_DIR}/RACE/ranking/${split}_qrels_oracle_context_pr.txt \
#     --rel_threshold 3 \
#     --n_questions 10 \
#     --passage_path outputs \
#     --tag ${aug_method} > ${split}.table1
# split=testb
# python3 -m evaluation.llm_prejudge \
#     --generator_name meta-llama/Meta-Llama-3.1-8B-Instruct \
#     --judgement_file judgements/${split}_oracle-report_judgements.jsonl \
#     --threshold 3 \
#     --qrels ${DATASET_DIR}/RACE/ranking/${split}_qrels_oracle_context_pr.txt \
#     --rel_threshold 3 \
#     --n_questions 15 \
#     --passage_path outputs \
#     --tag ${aug_method} > ${split}.table1
#
# aug_method=oracle-passages
# for rel_threshold in 1 3;do
#     split=test
#     python3 -m evaluation.llm_prejudge \
#         --generator_name meta-llama/Meta-Llama-3.1-8B-Instruct \
#         --threshold 3 \
#         --qrels ${DATASET_DIR}/RACE/ranking/${split}_qrels_oracle_context_pr.txt \
#         --rel_threshold ${rel_threshold} \
#         --n_questions 10 \
#         --passage_path ${DATASET_DIR}/RACE/passages \
#         --judgement_file ${DATASET_DIR}/RACE/ranking/${split}_judgements.jsonl \
#         --tag ${aug_method}' (rel='${rel_threshold}')'  >> ${split}.table1
#     split=testb
#     python3 -m evaluation.llm_prejudge \
#         --generator_name meta-llama/Meta-Llama-3.1-8B-Instruct \
#         --threshold 3 \
#         --qrels ${DATASET_DIR}/RACE/ranking/${split}_qrels_oracle_context_pr.txt \
#         --rel_threshold ${rel_threshold} \
#         --n_questions 15 \
#         --passage_path ${DATASET_DIR}/RACE/passages \
#         --judgement_file ${DATASET_DIR}/RACE/ranking/${split}_judgements.jsonl \
#         --tag ${aug_method}' (rel='${rel_threshold}')' >> ${split}.table1
# done
#
# # Table 1
# aug_method=vanilla
# for topk in 10 20 30;do
# for retriever in bm25 contriever splade;do
#     split=test
#     python3 -m evaluation.llm_prejudge \
#         --generator_name meta-llama/Meta-Llama-3.1-8B-Instruct \
#         --judgement_file ${DATASET_DIR}/RACE/ranking/${split}_judgements.jsonl \
#         --threshold 3 \
#         --run_file runs/baseline.${retriever}.race-${split}.passages.run \
#         --topk ${topk} \
#         --qrels ${DATASET_DIR}/RACE/ranking/${split}_qrels_oracle_context_pr.txt \
#         --n_questions 10 \
#         --passage_path ${DATASET_DIR}/RACE/passages \
#         --tag ${retriever}-${topk}-${aug_method} >> ${split}.table1
#     split=testb
#     python3 -m evaluation.llm_prejudge \
#         --generator_name meta-llama/Meta-Llama-3.1-8B-Instruct \
#         --judgement_file ${DATASET_DIR}/RACE/ranking/${split}_judgements.jsonl \
#         --threshold 3 \
#         --run_file runs/baseline.${retriever}.race-${split}.passages.run \
#         --topk ${topk} \
#         --qrels ${DATASET_DIR}/RACE/ranking/${split}_qrels_oracle_context_pr.txt \
#         --n_questions 15 \
#         --passage_path ${DATASET_DIR}/RACE/passages \
#         --tag ${retriever}-${topk}-${aug_method} >> ${split}.table1
# done
# done
#
# # [Bartsum]
# aug_method=bartsum
# topk=10
# for retriever in bm25 contriever splade;do
#     split=test
#     python3 -m evaluation.llm_prejudge \
#         --generator_name meta-llama/Meta-Llama-3.1-8B-Instruct \
#         --judgement_file judgements/${split}_${aug_method}_judgements.jsonl \
#         --threshold 3 \
#         --run_file runs/baseline.${retriever}.race-${split}.passages.run \
#         --topk ${topk} \
#         --qrels ${DATASET_DIR}/RACE/ranking/${split}_qrels_oracle_context_pr.txt \
#         --n_questions 10 \
#         --passage_path outputs/${split}_${aug_method}_psgs.jsonl \
#         --tag ${retriever}-${topk}-${aug_method} >> ${split}.table1
#     split=testb
#     python3 -m evaluation.llm_prejudge \
#         --generator_name meta-llama/Meta-Llama-3.1-8B-Instruct \
#         --judgement_file judgements/${split}_${aug_method}_judgements.jsonl \
#         --threshold 3 \
#         --run_file runs/baseline.${retriever}.race-${split}.passages.run \
#         --topk ${topk} \
#         --qrels ${DATASET_DIR}/RACE/ranking/${split}_qrels_oracle_context_pr.txt \
#         --n_questions 15 \
#         --passage_path outputs/${split}_${aug_method}_psgs.jsonl \
#         --tag ${retriever}-${topk}-${aug_method} >> ${split}.table1
# done
#
# # [Recomp]
# aug_method=recomp
# topk=10
# for retriever in bm25 contriever splade;do
#     split=test
#     python3 -m evaluation.llm_prejudge \
#         --generator_name meta-llama/Meta-Llama-3.1-8B-Instruct \
#         --judgement_file judgements/${split}_${aug_method}_judgements.jsonl \
#         --threshold 3 \
#         --run_file runs/baseline.${retriever}.race-${split}.passages.run \
#         --topk ${topk} \
#         --qrels ${DATASET_DIR}/RACE/ranking/${split}_qrels_oracle_context_pr.txt \
#         --n_questions 10 \
#         --passage_path outputs/${split}_${aug_method}_psgs.jsonl \
#         --tag ${retriever}-${topk}-${aug_method} >> ${split}.table1
#     split=testb
#     python3 -m evaluation.llm_prejudge \
#         --generator_name meta-llama/Meta-Llama-3.1-8B-Instruct \
#         --judgement_file judgements/${split}_${aug_method}_judgements.jsonl \
#         --threshold 3 \
#         --run_file runs/baseline.${retriever}.race-${split}.passages.run \
#         --topk ${topk} \
#         --qrels ${DATASET_DIR}/RACE/ranking/${split}_qrels_oracle_context_pr.txt \
#         --n_questions 15 \
#         --passage_path outputs/${split}_${aug_method}_psgs.jsonl \
#         --tag ${retriever}-${topk}-${aug_method} >> ${split}.table1
# done

# [Recomp]
# aug_method=recomp
# topk=10
# for retriever in bm25 contriever splade;do
#     split=test
#     python3 -m evaluation.llm_prejudge \
#         --generator_name meta-llama/Meta-Llama-3.1-8B-Instruct \
#         --judgement_file judgements/${split}_${aug_method}_judgements.jsonl \
#         --threshold 3 \
#         --run_file runs/baseline.${retriever}.race-${split}.passages.run \
#         --topk ${topk} \
#         --qrels ${DATASET_DIR}/RACE/ranking/${split}_qrels_oracle_context_pr.txt \
#         --n_questions 10 \
#         --passage_path outputs/${split}_${aug_method}_psgs.jsonl \
#         --tag ${retriever}-${topk}-${aug_method} >> ${split}.table1
#     split=testb
#     python3 -m evaluation.llm_prejudge \
#         --generator_name meta-llama/Meta-Llama-3.1-8B-Instruct \
#         --judgement_file judgements/${split}_${aug_method}_judgements.jsonl \
#         --threshold 3 \
#         --run_file runs/baseline.${retriever}.race-${split}.passages.run \
#         --topk ${topk} \
#         --qrels ${DATASET_DIR}/RACE/ranking/${split}_qrels_oracle_context_pr.txt \
#         --n_questions 15 \
#         --passage_path outputs/${split}_${aug_method}_psgs.jsonl \
#         --tag ${retriever}-${topk}-${aug_method} >> ${split}.table1
# done
