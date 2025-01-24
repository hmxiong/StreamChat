#!/usr/bin/env bash
source ~/anaconda3/etc/profile.d/conda.sh
conda activate freeva

CUDA_VISIBLE_DEVICES="0,1"
gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -ra GPULIST <<< "$gpu_list"
CHUNKS=${#GPULIST[@]}

# CUDA_VISIBLE_DEVICES=0 python video_data_filter.py  \
#                     --model_path /13390024681/All_Model_Zoo/LongVA-7B-DPO \
#                     --video_dir /13390024681/All_Data/EgoSchema/good_clips_git \
#                     --video_path_list /13390024681/All_Data/data/egoschema/fullset_anno.json \
#                     --output_dir /13390024681/llama/EfficientVideo/Ours/tools \
#                     --output_name EgoSchema_filtered

for IDX in $(seq 0 $((CHUNKS-1))); do
  CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python video_data_filter.py \
    --model_path /13390024681/All_Model_Zoo/LongVA-7B-DPO \
    --video_dir /13390024681/All_Data/EgoSchema/good_clips_git \
    --video_path_list /13390024681/All_Data/data/egoschema/fullset_anno.json \
    --output_dir /13390024681/llama/EfficientVideo/Ours/tools/Ego_Class \
    --output_name ${CHUNKS}_${IDX} \
    --num_chunks $CHUNKS \
    --chunk_idx $IDX &
done

wait

output_file=/13390024681/llama/EfficientVideo/Ours/tools/Ego_Class/merge.jsonl

# Clear out the output file if it exists.
> "$output_file"

# Loop through the indices and concatenate each file.
for IDX in $(seq 0 $((CHUNKS-1))); do
    cat /13390024681/llama/EfficientVideo/Ours/tools/Ego_Class/${CHUNKS}_${IDX}.json >> "$output_file"
done