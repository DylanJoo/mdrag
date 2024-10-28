#!/bin/sh
#SBATCH --job-name=contriever
#SBATCH --partition gpu
#SBATCH --gres=gpu:nvidia_rtx_a6000:1
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

RETRIEVER=facebook/contriever-msmarco

# contriever encode + indexing
# contriever search
for split in testb test;do
    # index
    # python -m pyserini.encode \
    #     input   --corpus ${DATASET_DIR}/RACE/passages/${split} \
    #             --fields contents \
    #             --delimiter "\n" \
    #     output  --embeddings ${INDEX_DIR}/RACE/contriever.race-${split}.passages.flat.faiss \
    #             --to-faiss \
    #     encoder --encoder ${RETRIEVER} \
    #             --encoder-class contriever \
    #             --fields contents \
    #             --batch 128 \
    #             --fp16

    # search
    python3 -m pyserini.search.faiss \
        --threads 16 --batch-size 128 \
        --encoder-class contriever \
        --encoder facebook/contriever-msmarco \
        --index ${INDEX_DIR}/RACE/contriever.race-${split}.passages.flat.faiss\
        --topics ${DATASET_DIR}/RACE/ranking_1/${split}_topics_report_request.tsv \
        --output runs/baseline.contriever.race-${split}.passages.run \
        --hits 100 
done
