#!/bin/sh
#SBATCH --job-name=rerank-mono
#SBATCH --partition gpu
#SBATCH --gres=gpu:3
#SBATCH --mem=32G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x.%j.out

# Set-up the environment.
source ${HOME}/.bashrc
conda activate rag
cd ~/mdrag

python -m reranking \
    --model_class monoT5 \
    --reranker_name_or_path castorini/monot5-3b-msmarco-10k \
    --tokenizer_name castorini/monot5-3b-msmarco-10k \
    --corpus ${DATASET_DIR}/mdrag-5K/passages \
    --topic ${DATASET_DIR}/mdrag-5K/ranking/test_topics_report_request.tsv \
    --batch_size 32  \
    --max_length 512 \
    --input_run retrieval/sample.run \
    --output reranking/baseline.bm25+monot5.mdrag-5K-test.passages.run \
    --top_k 100 \
    --device cuda \
    --fp16
