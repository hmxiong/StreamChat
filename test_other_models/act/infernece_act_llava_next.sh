#!/usr/bin/env bash
source ~/anaconda3/etc/profile.d/conda.sh
conda activate freeva
cd /13390024681/llama/EfficientVideo/Ours

CKPT_NAME="LLaVA-NExT-7B"
model_path="/13390024681/All_Model_Zoo/LLaVA-NExT-LLaMA3-8B"
cache_dir="./cache_dir"
GPT_Zero_Shot_QA="/13390024681/All_Data"
video_dir="${GPT_Zero_Shot_QA}/ActNet-Video/all_test"
gt_file_question="${GPT_Zero_Shot_QA}/ActiveNet-json/actnet_test.json"
# gt_file_answers="${GPT_Zero_Shot_QA}/msvd/annotations/test_a.json"
output_dir="/13390024681/llama/EfficientVideo/Ours/output/Act/${CKPT_NAME}"
# 150 3h

gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -ra GPULIST <<< "$gpu_list"

CHUNKS=${#GPULIST[@]}


for IDX in $(seq 0 $((CHUNKS-1))); do
  CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python3 /13390024681/llama/EfficientVideo/Ours/test_other_models/act/inference_act_llava_next.py \
      --model_path ${model_path} \
      --cache_dir ${cache_dir} \
      --video_dir ${video_dir} \
      --gt_file_question ${gt_file_question} \
      --output_dir ${output_dir} \
      --output_name ${CHUNKS}_${IDX} \
      --num_chunks $CHUNKS \
      --chunk_idx $IDX &
done

wait

output_file=${output_dir}/merge.jsonl

# Clear out the output file if it exists.
> "$output_file"

# Loop through the indices and concatenate each file.
for IDX in $(seq 0 $((CHUNKS-1))); do
    cat ${output_dir}/${CHUNKS}_${IDX}.json >> "$output_file"
done