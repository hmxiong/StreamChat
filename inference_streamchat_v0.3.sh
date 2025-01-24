#!/usr/bin/env bash
source ~/anaconda3/etc/profile.d/conda.sh
conda activate freeva
cd /13390024681/llama/EfficientVideo/Ours
# 218
yes | pip install json_lines

OURS_EGO=/13390024681/llama/EfficientVideo/Ours/output/result_test_streaming_fast_v0.3_0.2rate_chunk_size_40.json
LLAMA_3=/13390024681/All_Model_Zoo/llama3-8b-instruct-hf
SAVE_DIR=/13390024681/llama/EfficientVideo/All_Score

CUDA_VISIBLE_DEVICES=0,1 python inference_streaming_longva_offline.py \
                    --model_name /13390024681/All_Model_Zoo/LongVA-7B-DPO \
                    --video_dir /13390024681/All_Data/Streaming_Bench_v0.3 \
                    --annotations /13390024681/llama/EfficientVideo/Ours/streaming_bench_v0.3.json \
                    --conv-mode qwen_1_5  \
                    --temperature 0.2  \
                    --sample_rate 0.2 \
                    --chunk_size 40 \
                    --num_clusters 5 \
                    --interval 10 \
                    --short_window 20 \
                    --remember_window 5 \
                    --tau 5 \
                    --compress_rate 1 \
                    --num_beams 1 \
                    --mode on_line \
                    --memory_basic_dir /13390024681/llama/EfficientVideo/Ours/memory_bank/memories/streamingbench_chunk_size_40 \
                    --memory_file updata_memories_for_test_streaming_fast_v0.3_0.2rate_chunk_size_40.json \
                    --save_file ${OURS_EGO} \
                    --memory_search_top_k 1 \
                    --language en \
                    --multi_modal_memory 

CUDA_VISIBLE_DEVICES="0"
gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -ra GPULIST <<< "$gpu_list"
CHUNKS=${#GPULIST[@]}

mkdir ${SAVE_DIR}/StreamingBench_v0.3

################### Ours ###################
mkdir ${SAVE_DIR}/StreamingBench_v0.3/Ours_rate0.2_chunk40
for IDX in $(seq 0 $((CHUNKS-1))); do
  CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python3 /13390024681/llama/EfficientVideo/Ours/eval_video_qa_with_llama3_ours.py \
    --predict_file ${OURS_EGO} \
    --output_dir ${SAVE_DIR}/StreamingBench_v0.3/Ours_rate0.2_chunk40 \
    --output_name ${CHUNKS}_${IDX} \
    --llama3_path ${LLAMA_3} \
    --num_chunks $CHUNKS \
    --chunk_idx $IDX &
done

wait

output_file=${SAVE_DIR}/StreamingBench_v0.3/Ours_rate0.2_chunk40/streamingbench_merge.jsonl

# Clear out the output file if it exists.
> "$output_file"

# Loop through the indices and concatenate each file.
for IDX in $(seq 0 $((CHUNKS-1))); do
    cat ${SAVE_DIR}/StreamingBench_v0.3/Ours_rate0.2_chunk40/${CHUNKS}_${IDX}.json >> "$output_file"
done

python /13390024681/llama/EfficientVideo/Ours/calculate_score.py \
    --output_dir ${SAVE_DIR}/StreamingBench_v0.3/Ours_rate0.2_chunk40 \
    --output_name streamingbench_merge
