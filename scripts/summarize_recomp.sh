#!/bin/sh
# The following lines instruct Slurm to allocate one GPU.
#SBATCH --job-name=recomp
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
# for split in test testb;do
for split in test;do
    for retriever in bm25 contriever splade;do
        python3 summarize_ind.py \
            --model_name_or_path fangyuan/nq_abstractive_compressor \
            --model_class seq2seq \
            --template 'Question: \n Document: {P}\n Summary: ' \
            --batch_size 128 \
            --topk 30 \
            --max_length 512 \
            --topics ${DATASET_DIR}/RACE/ranking_1/${split}_topics_report_request.tsv \
            --collection ${DATASET_DIR}/RACE/passages \
            --run retrieval/baseline.${retriever}.race-${split}.passages.run \
            --output_file outputs/race-${split}-${retriever}-top30-recomp.jsonl \
            --truncate 5
    done
done
