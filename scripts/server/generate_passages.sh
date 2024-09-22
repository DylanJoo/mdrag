# Set-up the environment.
cd ~/mdrag

DATASET_DIR=/project/project_465001339/datasets

# Start the experiment.
# for shard_i in $(seq 0 0);do
#     srun --jobid=8033161 python3 augmentation/gen_passages.py \
#         --shard $shard_i --shard_size 2000 \
#         --config configs/mds-decontextualize.llama3-70b.yaml \
#         --split train \
#         --model meta-llama/Meta-Llama-3.1-70B-Instruct \
#         --model_tag metallama3.1-70b \
#         --tag psgs-gen \
#         --load_mode api \
#         --temperature 0.7 \
#         --max_new_tokens 640 \
#         --output_dir ${DATASET_DIR}/mdrag/shard_data/ \
#         --ampere_gpu 
# done

# test set
for shard_i in $(seq 2 4);do
    python3 augmentation/gen_passages.py \
        --shard $shard_i --shard_size 1000 \
        --config configs/mds-decontextualize.llama3-70b.yaml \
        --split test \
        --model meta-llama/Meta-Llama-3.1-70B-Instruct \
        --model_tag metallama3.1-70b \
        --tag psgs-gen \
        --load_mode api \
        --temperature 0.7 \
        --max_new_tokens 640 \
        --output_dir ${DATASET_DIR}/mdrag/shard_data/ \
        --ampere_gpu 
done
