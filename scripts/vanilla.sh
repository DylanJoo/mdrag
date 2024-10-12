#!/bin/sh
# The following lines instruct Slurm 
#SBATCH --job-name=eval-know
#SBATCH --cpus-per-task=16
#SBATCH --nodes=1
#SBATCH --mem=16G
#SBATCH --ntasks-per-node=1
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x-%j.out

# Set-up the environment.
source ${HOME}/.bashrc
conda activate rag
cd ~/mdrag

# Start the experiment.
for split in test testb;do
    for retriever in bm25 contriever;do
        python3 vanilla.py \
            --topics ${DATASET_DIR}/RACE/ranking/${split}_topics_report_request.tsv \
            --collection ${DATASET_DIR}/RACE/passages \
            --run retrieval/baseline.${retriever}.race-${split}.passages.run \
            --topk 10 \
            --output_file outputs/race-${split}-${retriever}-top10-vanilla.jsonl
    done
done

