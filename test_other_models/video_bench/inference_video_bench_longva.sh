#!/usr/bin/env bash
echo "dedd"
source ~/anaconda3/etc/profile.d/conda.sh
conda activate freeva
cd /13390024681/llama/EfficientVideo/Ours
echo "ded11d"
num_frame=$1
echo "ded11d"
CUDA_VISIBLE_DEVICES="0"
gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -ra GPULIST <<< "$gpu_list"

CHUNKS=${#GPULIST[@]}
echo "ded11d"
for IDX in $(seq 0 $((CHUNKS-1))); do
  echo "ded11d"
  CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python3 test_other_models/video_bench/inference_video_bench_longva.py \
      --model_path "/13390024681/All_Model_Zoo/LongVA-7B-DPO" \
      --num_chunks $CHUNKS \
      --num_frame $num_frame \
      --chunk_idx $IDX &
done

wait

# output_file=${output_dir}/streamingbench_merge.jsonl

# # Clear out the output file if it exists.
# > "$output_file"

# # Loop through the indices and concatenate each file.
# for IDX in $(seq 0 $((CHUNKS-1))); do
#     cat ${output_dir}/${CHUNKS}_${IDX}.json >> "$output_file"
# done