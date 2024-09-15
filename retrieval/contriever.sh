#!/bin/sh
#SBATCH --job-name=contriever-retrieval
#SBATCH --partition gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x.%j.out

# Set-up the environment.
source ${HOME}/.bashrc
conda activate rag
cd ~/mdrag

# pyserini==0.24.0 (add coniditons for collection uses 'contents')
# contriever encode with faiss
python -m pyserini.encode \
    input   --corpus ${DATASET_DIR}/mdrag-5K/passages \
            --fields contents \
            --delimiter "\n" \
    output  --embeddings ${INDEX_DIR}/mdrag-5K-passages.contriever \
            --to-faiss \
    encoder --encoder facebook/contriever-msmarco \
            --encoder-class contriever \
            --fields contents \
            --batch 32 \
            --fp16

# contriever search
for split in train test;do
    python3 -m pyserini.search.faiss \
        --threads 16 --batch-size 64 \
        --encoder-class contriever \
        --encoder facebook/contriever-msmarco \
        --index ${INDEX_DIR}/mdrag-5K-passages.contriever \
        --topics ${DATASET_DIR}/mdrag-5K/ranking/${split}_topics_report_request.tsv \
        --output retrieval/baseline.contriever.mdrag-5K-${split}.passages.run \
        --hits 100 
done
