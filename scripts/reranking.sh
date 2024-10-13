#!/bin/sh
#SBATCH --job-name=rerank
#SBATCH --partition gpu
#SBATCH --gres=gpu:nvidia_titan_v:2
#SBATCH --mem=32G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x-%j.out

# Set-up the environment.
source ${HOME}/.bashrc
conda activate rag
cd ~/mdrag

for split in test testb;do
    model_name=castorini/monot5-3b-msmarco-10k
    model_class=monoT5
    python -m reranking \
        --model_class ${model_class} \
        --reranker_name_or_path ${model_name} \
        --tokenizer_name ${model_name} \
        --corpus ${DATASET_DIR}/RACE/passages \
        --topic ${DATASET_DIR}/RACE/ranking/${split}_topics_report_request.tsv \
        --batch_size 16 \
        --max_length 512 \
        --input_run retrieval/baseline.bm25.race-test.passages.run \
        --output reranking/reranking.bm25+${model_class}.race-${split}.passages.run \
        --top_k 100 \
        --device auto \
        --fp16

    # model_name=cross-encoder/ms-marco-MiniLM-L-12-v2 
    # model_class=monoBERT
    # python -m reranking \
    #     --model_class ${model_class} \
    #     --reranker_name_or_path ${model_name} \
    #     --tokenizer_name ${model_name} \
    #     --corpus ${DATASET_DIR}/RACE/passages \
    #     --topic ${DATASET_DIR}/RACE/ranking/${split}_topics_report_request.tsv \
    #     --batch_size 128  \
    #     --max_length 512 \
    #     --input_run retrieval/baseline.bm25.race-test.passages.run \
    #     --output reranking/reranking.bm25+${model_class}.race-${split}.passages.run \
    #     --top_k 100 \
    #     --device cuda \
    #     --fp16
done
