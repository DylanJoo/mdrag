#!/bin/sh
# The following lines instruct Slurm to allocate one GPU.
#SBATCH --job-name=bart-sum
#SBATCH --partition gpu
#SBATCH --gres=gpu:nvidia_rtx_a6000:1
#SBATCH --mem=16G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=72:00:00
#SBATCH --output=logs/%x.%j.out

# Set-up the environment.
source ${HOME}/.bashrc
conda activate rag
cd ~/mdrag

# Start the experiment.
for split in testb;do
    for retriever in bm25+monoT5 contriever+monoT5 splade+monoT5;do
        # rerank + vanilla
        python3 vanilla.py \
            --topics ${DATASET_DIR}/RACE/ranking_1/${split}_topics_report_request.tsv \
            --collection ${DATASET_DIR}/RACE/passages \
            --run reranking/reranking.${retriever}.race-${split}.passages.run \
            --topk 30 \
            --output_file outputs/race-${split}-${retriever}-top30-vanilla.jsonl

        # rerank + summarize 
        python3 summarize_ind.py \
            --model_name_or_path facebook/bart-large-cnn \
            --model_class seq2seq \
            --template '{P}' \
            --batch_size 32 \
            --topk 30 \
            --max_length 512 \
            --topics ${DATASET_DIR}/RACE/ranking_1/${split}_topics_report_request.tsv \
            --collection ${DATASET_DIR}/RACE/passages \
            --run reranking/reranking.${retriever}.race-${split}.passages.run \
            --output_file outputs/race-${split}-${retriever}-top30-bartsum.jsonl
    done
done

