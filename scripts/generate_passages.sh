#!/bin/sh
# The following lines instruct Slurm to allocate one GPU.
#SBATCH --job-name=psg-gen
#SBATCH --partition gpu
#SBATCH --gres=gpu:nvidia_l40:1
#SBATCH --mem=32G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=120:00:00
#SBATCH --output=logs/%x.%j.out

# Set-up the environment.
source ${HOME}/.bashrc
conda activate rag
cd ~/mdrag

# Start the experiment.
for shard_i in $(seq 0 0);do
    python3 augmentation/gen_passages.py \
        --shard $shard_i --shard_size 200 \
        --config configs/mds-decontextualize.llama3-8b-chat.yaml \
        --split train \
        --model_tag metallama3.1-8b \
        --tag psgs-gen \
        --load_mode vllm \
        --temperature 0.7 \
        --max_new_tokens 640 \
        --quick_test 5000 \
        --output_dir ${DATASET_DIR}/mdrag-5K/shard_data/ \
        --ampere_gpu 
done
