#!/bin/sh
# The following lines instruct Slurm to allocate one GPU.
#SBATCH --job-name=recomp
#SBATCH --partition gpu
#SBATCH --gres=gpu:tesla_p40:1
#SBATCH --mem=16G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=15:00:00
#SBATCH --output=logs/%x.%j.out

# Set-up the environment.
source ${HOME}/.bashrc
conda activate rag
cd ~/mdrag

# Start the experiment.

for split in test;do
    python3 summarize_ind.py \
        --model_name_or_path fangyuan/nq_abstractive_compressor \
        --model_class seq2seq \
        --template 'Question: {Q}\n Document: {P}\n Summary: ' \
        --batch_size 32 \
        --topk 10 \
        --max_length 1024 \
        --topics ${DATASET_DIR}/mdrag-5K/ranking/${split}_topics_report_request.tsv \
        --collection ${DATASET_DIR}/mdrag-5K/passages \
        --run retrieval/baseline.bm25.mdrag-5K-${split}.passages.run \
        --output_file outputs/mdrag-5K-${split}-bm25-top10-recomp.jsonl \
        --truncate
done

