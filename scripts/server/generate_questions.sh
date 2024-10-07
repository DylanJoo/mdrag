# Set-up the environment.
cd ~/mdrag

DATASET_DIR=/project/project_465001339/datasets

# Start the experiment.
# for shard_i in $(seq 0 24);do
#     python3 augmentation/gen_questions.py \
#         --shard $shard_i --shard_size 200 \
#         --config configs/mds-decontextualize.llama3-70b.yaml \
#         --split test \
#         --model meta-llama/Meta-Llama-3.1-70B-Instruct \
#         --model_tag metallama3.1-70b \
#         --tag ques-gen \
#         --load_mode api \
#         --temperature 0.7 \
#         --max_new_tokens 640 \
#         --output_dir ${DATASET_DIR}/mdrag/shard_data/ \
#         --ampere_gpu
# done

python3 augmentation/gen_questions.py \
    --shard 0 \
    --duc04_file /project/project_465001339/datasets/duc04 \
    --config configs/mds-decontextualize.llama3-70b.yaml \
    --split testb \
    --model meta-llama/Meta-Llama-3.1-70B-Instruct \
    --model_tag metallama3.1-70b \
    --tag ques-gen \
    --load_mode api \
    --temperature 0.7 \
    --max_new_tokens 640 \
    --output_dir ${DATASET_DIR}/mdrag/shard_data/ \
    --ampere_gpu
