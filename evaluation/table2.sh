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

TAU=3

# [Oracle]
aug_method=oracle-report

for split in test testb;do
python3 -m evaluation.llm_prejudge \
    --generator_name meta-llama/Meta-Llama-3.1-70B-Instruct \
    --dataset_dir ${DATASET_DIR}/crux \
    --rel_subset 3 \
    --split ${split} \
    --threshold ${TAU} \
    --weighted_factor 0.25 \
    --passage_path ${DATASET_DIR}/crux/outputs/${split}_${aug_method}_psgs.jsonl \
    --judgement_file ${DATASET_DIR}/crux/judgements/${split}_${aug_method}_judgements.jsonl \
    --tag ${aug_method} > ${split}.table2
done

# aug_method=oracle-passages
# for rel_subset in 1 3;do
#     split=test
#     python3 -m evaluation.llm_prejudge \
#         --generator_name meta-llama/Meta-Llama-3.1-70B-Instruct \
#         --threshold ${TAU} \
#         --rel_subset ${rel_subset} \
#         --split ${split} \
#         --passage_path ${DATASET_DIR}/crux/passages \
#         --judgement_file ${DATASET_DIR}/crux/ranking_${TAU}/${split}_judgements.jsonl \
#         --tag ${aug_method}' (rel='${rel_subset}')'  >> ${split}.table2
#     split=testb
#     python3 -m evaluation.llm_prejudge \
#         --generator_name meta-llama/Meta-Llama-3.1-70B-Instruct \
#         --threshold ${TAU} \
#         --rel_subset ${rel_subset} \
#         --split ${split} \
#         --passage_path ${DATASET_DIR}/crux/passages \
#         --judgement_file ${DATASET_DIR}/crux/ranking_${TAU}/${split}_judgements.jsonl \
#         --tag ${aug_method}' (rel='${rel_subset}')' >> ${split}.table2
# done
#
# # [vanilla]
# aug_method=vanilla
# topk=10
# for retriever in bm25 contriever splade;do
#     split=test
#     python3 -m evaluation.llm_prejudge \
#         --generator_name meta-llama/Meta-Llama-3.1-70B-Instruct \
#         --judgement_file ${DATASET_DIR}/crux/ranking_${TAU}/${split}_judgements.jsonl \
#         --threshold ${TAU} \
#         --run_file runs/baseline.${retriever}.race-${split}.passages.run \
#         --topk ${topk} \
#         --split ${split} \
#         --passage_path ${DATASET_DIR}/crux/passages \
#         --tag ${retriever}-${topk}-${aug_method} >> ${split}.table2
#     split=testb
#     python3 -m evaluation.llm_prejudge \
#         --generator_name meta-llama/Meta-Llama-3.1-70B-Instruct \
#         --judgement_file ${DATASET_DIR}/crux/ranking_${TAU}/${split}_judgements.jsonl \
#         --threshold ${TAU} \
#         --run_file runs/baseline.${retriever}.race-${split}.passages.run \
#         --topk ${topk} \
#         --split ${split} \
#         --passage_path ${DATASET_DIR}/crux/passages \
#         --tag ${retriever}-${topk}-${aug_method} >> ${split}.table2
# done
#
# # [Bartsum] [recomp]
# topk=10
# for aug_method in bartsum recomp;do
# for retriever in bm25 contriever splade;do
#     split=test
#     python3 -m evaluation.llm_prejudge \
#         --generator_name meta-llama/Meta-Llama-3.1-70B-Instruct \
#         --judgement_file judgements/${split}_${aug_method}_judgements.jsonl \
#         --threshold ${TAU} \
#         --run_file runs/baseline.${retriever}.race-${split}.passages.run \
#         --topk ${topk} \
#         --passage_path outputs/${split}_${aug_method}_psgs.jsonl \
#         --tag ${retriever}-${topk}-${aug_method} >> ${split}.table2
#     split=testb
#     python3 -m evaluation.llm_prejudge \
#         --generator_name meta-llama/Meta-Llama-3.1-70B-Instruct \
#         --judgement_file judgements/${split}_${aug_method}_judgements.jsonl \
#         --threshold ${TAU} \
#         --run_file runs/baseline.${retriever}.race-${split}.passages.run \
#         --topk ${topk} \
#         --passage_path outputs/${split}_${aug_method}_psgs.jsonl \
#         --tag ${retriever}-${topk}-${aug_method} >> ${split}.table2
# done
# done
