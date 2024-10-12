#!/bin/sh
#SBATCH --job-name=splade
#SBATCH --partition gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --nodes=1
#SBATCH --time=72:00:00
#SBATCH --output=logs/%x-%j.out

# Set-up the environment.
source ${HOME}/.bashrc
conda activate rag
cd ~/mdrag

RETRIEVER=naver/splade-v3

# Encode 
python retrieval/mlm_encode.py \
    --model_name_or_path ${RETRIEVER} \
    --tokenizer_name ${RETRIEVER} \
    --collection_dir ${DATASET_DIR}/RACE/passages \
    --collection_output ${INDEX_DIR}/RACE/splade-v3.race-passages.encoded/vectors.jsonl \
    --batch_size 32 \
    --max_length 256 \
    --quantization_factor 100

# Index (cpu only)
python -m pyserini.index.lucene \
  --collection JsonVectorCollection \
  --input ${INDEX_DIR}/RACE/splade-v3.race-passages.encoded \
  --index ${INDEX_DIR}/RACE/splade-v3.race-passages.lucene \
  --generator DefaultLuceneDocumentGenerator \
  --threads 36 \
  --impact --pretokenized

# Search
for split in test testb;do
    python -m pyserini.search.lucene \
        --index ${INDEX_DIR}/RACE/splade-v3.race-passages.lucene \
        --topics ${DATASET_DIR}/RACE/ranking/${split}_topics_report_request.tsv \
	    --encoder ${RETRIEVER} \
        --output retrieval/baseline.splade.race-${split}.passages.run \
	    --output-format trec \
	    --batch 36 --threads 12 \
	    --hits 1000 \
	    --impact
done
