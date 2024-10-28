#!/bin/sh
# The following lines instruct Slurm to allocate one GPU.
#SBATCH --job-name=zscomp
#SBATCH --partition gpu
#SBATCH --gres=gpu:nvidia_rtx_a6000:1
#SBATCH --mem=16G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x.%j.out

# Set-up the environment.
source ${HOME}/.bashrc
conda activate rag
cd ~/mdrag

# Start the experiment.
for split in testb test;do
    for retriever in bm25 contriever splade;do
        python3 summarize_ind.py \
            --model_name_or_path meta-llama/Llama-3.2-1B-Instruct \
            --model_class causalLM \
            --template 'Based on the given topic of user request, summarize the provided passage. Write the summary within `<s>` and `</s>` tags.\n\nTopic: {Q}\nPassage: {P}\nSummary: <s>' \
            --batch_size 128 \
            --topk 10 \
            --max_length 612 \
            --topics ${DATASET_DIR}/RACE/ranking/${split}_topics_report_request.tsv \
            --collection ${DATASET_DIR}/RACE/passages \
            --run retrieval/baseline.${retriever}.race-${split}.passages.run \
            --output_file outputs/race-${split}-${retriever}-top10-zscomp.jsonl \
            --truncate 40
    done
done
