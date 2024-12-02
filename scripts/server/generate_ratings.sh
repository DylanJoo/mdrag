# Set-up the environment.
cd ~/mdrag

DATASET_DIR=/project/project_465001396/dylan/datasets

# Start the experiment.
for shard_i in 0;do
    python3 augmentation/gen_ratings.py \
        --shard ${shard_i} --shard_size 10 \
        --multi_news_file /project/project_465001339/datasets/multi_news \
        --shard_dir ${DATASET_DIR}/mdrag/shard_data \
        --config configs/mds-decontextualize.llama3-70b.yaml \
        --split test \
        --model meta-llama/Meta-Llama-3.1-70B-Instruct \
        --model_tag metallama3.1-70b \
        --tag ratings-gen \
        --load_mode api \
        --temperature 0.7 \
        --max_new_tokens 2 \
        --output_dir ${DATASET_DIR}/mdrag/shard_data/ \
        --ampere_gpu
done

# python3 augmentation/gen_ratings.py \
#     --shard_dir ${DATASET_DIR}/mdrag/shard_data \
#     --config configs/mds-decontextualize.llama3-70b.yaml \
#     --split testb \
#     --model meta-llama/Meta-Llama-3.1-70B-Instruct \
#     --model_tag metallama3.1-70b \
#     --tag ratings-gen \
#     --load_mode api \
#     --temperature 0.7 \
#     --max_new_tokens 2 \
#     --output_dir ${DATASET_DIR}/mdrag/shard_data/ \
#     --n_questions 15 \
#     --ampere_gpu
