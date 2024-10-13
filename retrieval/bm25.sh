#!/bin/sh
# The following lines instruct Slurm 
#SBATCH --job-name=bm25
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
# for level in documents passages;do
#     python -m pyserini.index.lucene \
#         --collection JsonCollection \
#         --input ${DATASET_DIR}/RACE/${level} \
#         --index ${INDEX_DIR}/RACE/bm25.race-${level}.lucene \
#         --generator DefaultLuceneDocumentGenerator \
#         --threads 16
# done

# bm25 search
for split in test testb;do
    python -m pyserini.search.lucene \
      --index ${INDEX_DIR}/RACE/bm25.race-documents.lucene \
      --topics ${DATASET_DIR}/RACE/ranking/${split}_topics_report_request.tsv \
      --output retrieval/baseline.bm25.race-${split}.documents.run \
      --hits 100 \
      --bm25

    python -m pyserini.search.lucene \
      --index ${INDEX_DIR}/RACE/bm25.race-passages.lucene \
      --topics ${DATASET_DIR}/RACE/ranking/${split}_topics_report_request.tsv \
      --output retrieval/baseline.bm25.race-${split}.passages.run \
      --hits 1000 \
      --bm25
done
