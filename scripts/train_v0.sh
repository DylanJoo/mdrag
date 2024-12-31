#!/bin/sh
# The following lines instruct Slurm to allocate one GPU.
#SBATCH --job-name=train-crux
#SBATCH --partition gpu
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --ntasks-per-node=4
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x-%j.out

# Set-up the environment.
module load 2024
module load UCX-CUDA/1.12.1-GCCcore-11.3.0-CUDA-11.7.0

export CUDA_HOME=/sw/arch/RHEL8/EB_production/2022/software/CUDA/11.7.0
conda activate ledr

cd ~/mdrag

accelerate launch \
    --config_file configs/default_train.yaml \
    train.py \
    --model_class cepe \
    --model_name_or_path hyen/CEPED-LLaMA-2-Chat-7B \
    --tokenizer_name hyen/CEPED-LLaMA-2-Chat-7B \
    --config_name hyen/CEPED-LLaMA-2-Chat-7B \
    --output_dir models/ckpt/cepe-crux-tarin \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --gradient_checkpointing true \
    --train_file data/inverted-mds/mds-5k-greedy-1.jsonl \
    --max_src_length 1024 \
    --max_tgt_length 1024 \
    --learning_rate 1e-5 \
    --warmup_ratio 0.1 \
    --lr_scheduler_type cosine \
    --max_steps 10000 \
    --save_steps 1000 \
    --eval_steps 500 \
    --do_train --do_eval \
    --bf16 \
    --max_num_contexts 5 \
    --num_distractor_docs 2 \
    --num_redundant_docs 1 \
    --collator_type standard \
    --report_to wandb --run_name cepe-crux
