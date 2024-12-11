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
aug_method=oracle-report
split=test
python3 -m evaluation.llm_prejudge \
    --generator_name meta-llama/Meta-Llama-3.1-70B-Instruct \
    --weighted_factor 0.25 \
    --judgement_file judgements/${split}_oracle-report_judgements.jsonl \
    --threshold 3 \
    --qrels ${DATASET_DIR}/crux/ranking/${split}_qrels_oracle_context_pr.txt \
    --rel_threshold 3 \
    --passage_path outputs \
    --tag ${aug_method} > ${split}.table2
split=testb
python3 -m evaluation.llm_prejudge \
    --generator_name meta-llama/Meta-Llama-3.1-70B-Instruct \
    --judgement_file judgements/${split}_oracle-report_judgements.jsonl \
    --threshold 3 \
    --qrels ${DATASET_DIR}/crux/ranking/${split}_qrels_oracle_context_pr.txt \
    --rel_threshold 3 \
    --passage_path outputs \
    --tag ${aug_method} > ${split}.table2

aug_method=oracle-passages
for rel_threshold in 1 3;do
    split=test
    python3 -m evaluation.llm_prejudge \
        --generator_name meta-llama/Meta-Llama-3.1-70B-Instruct \
        --threshold 3 \
        --qrels ${DATASET_DIR}/crux/ranking/${split}_qrels_oracle_context_pr.txt \
        --rel_threshold ${rel_threshold} \
        --passage_path ${DATASET_DIR}/crux/passages \
        --judgement_file ${DATASET_DIR}/crux/ranking/${split}_judgements.jsonl \
        --tag ${aug_method}' (rel='${rel_threshold}')'  >> ${split}.table2
    split=testb
    python3 -m evaluation.llm_prejudge \
        --generator_name meta-llama/Meta-Llama-3.1-70B-Instruct \
        --threshold 3 \
        --qrels ${DATASET_DIR}/crux/ranking/${split}_qrels_oracle_context_pr.txt \
        --rel_threshold ${rel_threshold} \
        --passage_path ${DATASET_DIR}/crux/passages \
        --judgement_file ${DATASET_DIR}/crux/ranking/${split}_judgements.jsonl \
        --tag ${aug_method}' (rel='${rel_threshold}')' >> ${split}.table2
done

# [vanilla]
aug_method=vanilla
topk=10
for retriever in bm25 contriever splade;do
    split=test
    python3 -m evaluation.llm_prejudge \
        --generator_name meta-llama/Meta-Llama-3.1-70B-Instruct \
        --judgement_file ${DATASET_DIR}/crux/ranking/${split}_judgements.jsonl \
        --threshold 3 \
        --run_file runs/baseline.${retriever}.race-${split}.passages.run \
        --topk ${topk} \
        --qrels ${DATASET_DIR}/crux/ranking/${split}_qrels_oracle_context_pr.txt \
        --passage_path ${DATASET_DIR}/crux/passages \
        --tag ${retriever}-${topk}-${aug_method} >> ${split}.table2
    split=testb
    python3 -m evaluation.llm_prejudge \
        --generator_name meta-llama/Meta-Llama-3.1-70B-Instruct \
        --judgement_file ${DATASET_DIR}/crux/ranking/${split}_judgements.jsonl \
        --threshold 3 \
        --run_file runs/baseline.${retriever}.race-${split}.passages.run \
        --topk ${topk} \
        --qrels ${DATASET_DIR}/crux/ranking/${split}_qrels_oracle_context_pr.txt \
        --passage_path ${DATASET_DIR}/crux/passages \
        --tag ${retriever}-${topk}-${aug_method} >> ${split}.table2
done

# [Bartsum] [recomp]
topk=10
for aug_method in bartsum recomp;do
for retriever in bm25 contriever splade;do
    split=test
    python3 -m evaluation.llm_prejudge \
        --generator_name meta-llama/Meta-Llama-3.1-70B-Instruct \
        --judgement_file judgements/${split}_${aug_method}_judgements.jsonl \
        --threshold 3 \
        --run_file runs/baseline.${retriever}.race-${split}.passages.run \
        --topk ${topk} \
        --qrels ${DATASET_DIR}/crux/ranking/${split}_qrels_oracle_context_pr.txt \
        --passage_path outputs/${split}_${aug_method}_psgs.jsonl \
        --tag ${retriever}-${topk}-${aug_method} >> ${split}.table2
    split=testb
    python3 -m evaluation.llm_prejudge \
        --generator_name meta-llama/Meta-Llama-3.1-70B-Instruct \
        --judgement_file judgements/${split}_${aug_method}_judgements.jsonl \
        --threshold 3 \
        --run_file runs/baseline.${retriever}.race-${split}.passages.run \
        --topk ${topk} \
        --qrels ${DATASET_DIR}/crux/ranking/${split}_qrels_oracle_context_pr.txt \
        --passage_path outputs/${split}_${aug_method}_psgs.jsonl \
        --tag ${retriever}-${topk}-${aug_method} >> ${split}.table2
done
done
