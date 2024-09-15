#!/bin/sh
# The following lines instruct Slurm 
#SBATCH --job-name=bm25-retrieval
#SBATCH --cpus-per-task=32
#SBATCH --nodes=1
#SBATCH --mem=32G
#SBATCH --ntasks-per-node=1
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x-%j.out

# Set-up the environment.
source ${HOME}/.bashrc
conda activate rag
cd ~/mdrag

# bm25 indexing
# python -m pyserini.index.lucene \
#     --collection JsonCollection \
#     --input  ${DATASET_DIR}/mdrag-5K/documents \
#     --index ${INDEX_DIR}/mdrag-5K-documents.lucene \
#     --generator DefaultLuceneDocumentGenerator \
#     --threads 8
#
# python -m pyserini.index.lucene \
#     --collection JsonCollection \
#     --input  ${DATASET_DIR}/mdrag-5K/passages \
#     --index ${INDEX_DIR}/mdrag-5K-passages.lucene \
#     --generator DefaultLuceneDocumentGenerator \
#     --threads 8

# bm25 search
for split in train test;do
    python -m pyserini.search.lucene \
      --index ${INDEX_DIR}/mdrag-5K-documents.lucene \
      --topics ${DATASET_DIR}/mdrag-5K/ranking/${split}_topics_report_request.tsv \
      --output retrieval/baseline.bm25.mdrag-5K-${split}.documents.run \
      --hits 100 \
      --bm25

    python -m pyserini.search.lucene \
      --index ${INDEX_DIR}/mdrag-5K-passages.lucene \
      --topics ${DATASET_DIR}/mdrag-5K/ranking/${split}_topics_report_request.tsv \
      --output retrieval/baseline.bm25.mdrag-5K-${split}.passages.run \
      --hits 100 \
      --bm25
done
