#!/usr/bin/env bash
source ~/anaconda3/etc/profile.d/conda.sh
conda activate freeva
cd /13390024681/llama/EfficientVideo/Ours

# bash /13390024681/llama/EfficientVideo/Ours/test_other_models/streambench/inference_streambench_minicmp.sh
# bash /13390024681/llama/EfficientVideo/Ours/test_other_models/streambench/inference_streambench_intervl2.sh
# bash /13390024681/llama/EfficientVideo/Ours/test_other_models/streambench/inference_streambench_vila.sh
# bash /13390024681/llama/EfficientVideo/Ours/test_other_models/streambench/inference_streambench_xcp.sh
# bash /13390024681/llama/EfficientVideo/MovieChat/run_test_ego_streaming.sh

# CUDA_VISIBLE_DEVICES=0 bash /13390024681/llama/EfficientVideo/eval_ego_streaming.sh
yes | pip install json_lines

LLAMA_3=/13390024681/All_Model_Zoo/llama3-8b-instruct-hf
SAVE_DIR=/13390024681/llama/EfficientVideo/All_Score
MINICMP_OUT=/13390024681/llama/EfficientVideo/Ours/output/StreamingBench_v0.3/MiniCMP_v2_6/streamingbench_merge.jsonl
INTERVL_OUT=/13390024681/llama/EfficientVideo/Ours/output/StreamingBench_v0.3/InterVL_2/streamingbench_merge.jsonl
VILA_OUT=/13390024681/llama/EfficientVideo/Ours/output/StreamingBench_v0.3/VILA_8B/streamingbench_merge.jsonl
INTERVLXCMP_OUT=/13390024681/llama/EfficientVideo/Ours/output/StreamingBench_v0.3/InterLM_Xcomposer_2_6/streamingbench_merge.jsonl

CUDA_VISIBLE_DEVICES="0,1"
gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -ra GPULIST <<< "$gpu_list"
CHUNKS=${#GPULIST[@]}

################### intervl2 ###################
mkdir ${SAVE_DIR}/InterVL_2
for IDX in $(seq 0 $((CHUNKS-1))); do
  CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python3 /13390024681/llama/EfficientVideo/Ours/test_other_models/eval_video_qa_with_llama3_others.py \
    --predict_file ${INTERVL_OUT} \
    --output_dir ${SAVE_DIR}/InterVL_2 \
    --output_name ${CHUNKS}_${IDX} \
    --llama3_path ${LLAMA_3} \
    --num_chunks $CHUNKS \
    --chunk_idx $IDX &
done

wait

output_file=${SAVE_DIR}/InterVL_2/streamingbench_merge.jsonl

# Clear out the output file if it exists.
> "$output_file"

# Loop through the indices and concatenate each file.
for IDX in $(seq 0 $((CHUNKS-1))); do
    cat ${SAVE_DIR}/InterVL_2/${CHUNKS}_${IDX}.json >> "$output_file"
done

python /13390024681/llama/EfficientVideo/Ours/calculate_score.py \
    --output_dir /13390024681/llama/EfficientVideo/All_Score/InterVL_2 \
    --output_name streamingbench_merge
