#!/usr/bin/env bash
source ~/anaconda3/etc/profile.d/conda.sh
conda activate freeva
cd /13390024681/llama/EfficientVideo/Ours

############################# MSVD #########################
CKPT_NAME="LongVA-7B-DPO"
model_path="/13390024681/All_Model_Zoo/LongVA-7B-DPO"
cache_dir="./cache_dir"
GPT_Zero_Shot_QA="/13390024681/All_Data"
video_dir="${GPT_Zero_Shot_QA}/msvd/YouTubeClips"
gt_file_question="${GPT_Zero_Shot_QA}/msvd/annotations/qa_test.json"
gt_file_answers="${GPT_Zero_Shot_QA}/msvd/annotations/test_a.json"
output_dir="/13390024681/llama/EfficientVideo/Ours/output/MSVD_Zero_Shot_QA/${CKPT_NAME}"


gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -ra GPULIST <<< "$gpu_list"

CHUNKS=${#GPULIST[@]}


for IDX in $(seq 0 $((CHUNKS-1))); do
  CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python3 /13390024681/llama/EfficientVideo/Ours/test_other_models/infernece_video_qa_longva.py \
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

output_file=${output_dir}/msvd_merge.jsonl

# Clear out the output file if it exists.
> "$output_file"

# Loop through the indices and concatenate each file.
for IDX in $(seq 0 $((CHUNKS-1))); do
    cat ${output_dir}/${CHUNKS}_${IDX}.json >> "$output_file"
done

# ############################# MSRVTT #########################
# CKPT_NAME="LongVA-7B-DPO"
# model_path="/13390024681/All_Model_Zoo/LongVA-7B-DPO"
# cache_dir="./cache_dir"
# GPT_Zero_Shot_QA="/13390024681/All_Data"
# video_dir="${GPT_Zero_Shot_QA}/MSRVTT/TestVideo"
# gt_file_question="${GPT_Zero_Shot_QA}/MSRVTT/qa_test.json"
# gt_file_answers="${GPT_Zero_Shot_QA}/MSRVTT/test_a.json"
# output_dir="/13390024681/llama/EfficientVideo/Ours/output/MSRVTT_Zero_Shot_QA/${CKPT_NAME}"


# gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
# IFS=',' read -ra GPULIST <<< "$gpu_list"

# CHUNKS=${#GPULIST[@]}


# for IDX in $(seq 0 $((CHUNKS-1))); do
#   CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python3 /13390024681/llama/EfficientVideo/Ours/test_other_models/infernece_video_qa_longva.py \
#       --model_path ${model_path} \
#       --cache_dir ${cache_dir} \
#       --video_dir ${video_dir} \
#       --gt_file_question ${gt_file_question} \
#       --output_dir ${output_dir} \
#       --output_name ${CHUNKS}_${IDX} \
#       --num_chunks $CHUNKS \
#       --chunk_idx $IDX &
# done

# wait

# output_file=${output_dir}/msrvtt_merge.jsonl

# # Clear out the output file if it exists.
# > "$output_file"

# # Loop through the indices and concatenate each file.
# for IDX in $(seq 0 $((CHUNKS-1))); do
#     cat ${output_dir}/${CHUNKS}_${IDX}.json >> "$output_file"
# done

########################### eval and calculate score #########################
conda activate freeva

LLAMA_3=/13390024681/All_Model_Zoo/llama3-8b-instruct-hf
SAVE_DIR=/13390024681/llama/EfficientVideo/All_Score

LONGVA_MSVD=/13390024681/llama/EfficientVideo/Ours/output/MSVD_Zero_Shot_QA/LongVA-7B-DPO/msvd_merge.jsonl
LONGVA_MSRVTT=/13390024681/llama/EfficientVideo/Ours/output/MSRVTT_Zero_Shot_QA/LongVA-7B-DPO/msrvtt_merge.jsonl

# MOVIECHAT_MSVD=/13390024681/llama/EfficientVideo/MovieChat/output/moviechat_msvd.json
# MOVIECHAT_MSRVTT=/13390024681/llama/EfficientVideo/MovieChat/output/moviechat_msrvtt.json

# CUDA_VISIBLE_DEVICES="0,1"
gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -ra GPULIST <<< "$gpu_list"
CHUNKS=${#GPULIST[@]}

yes | pip install json_lines

################### MSVD ###################
###################################### LongVA

mkdir ${SAVE_DIR}/MSVD/LongVA
for IDX in $(seq 0 $((CHUNKS-1))); do
  CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python3 /13390024681/llama/EfficientVideo/FreeVA/scripts/gpt_eval/eval_video_qa_with_llama3.py \
    --predict_file ${LONGVA_MSVD} \
    --output_dir ${SAVE_DIR}/MSVD/LongVA \
    --output_name ${CHUNKS}_${IDX} \
    --llama3_path ${LLAMA_3} \
    --num_chunks $CHUNKS \
    --chunk_idx $IDX &
done

wait

output_file=${SAVE_DIR}/MSVD/LongVA/merge.jsonl

# Clear out the output file if it exists.
> "$output_file"

# Loop through the indices and concatenate each file.
for IDX in $(seq 0 $((CHUNKS-1))); do
    cat ${SAVE_DIR}/MSVD/LongVA/${CHUNKS}_${IDX}.json >> "$output_file"
done

python /13390024681/llama/EfficientVideo/FreeVA/scripts/gpt_eval/calculate_score.py \
    --output_dir ${SAVE_DIR}/MSVD/LongVA \
    --output_name merge

################### MSRVTT ###################
###################################### LongVA

# mkdir ${SAVE_DIR}/MSRVTT/LongVA
# for IDX in $(seq 0 $((CHUNKS-1))); do
#   CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python3 /13390024681/llama/EfficientVideo/FreeVA/scripts/gpt_eval/eval_video_qa_with_llama3.py \
#     --predict_file ${LONGVA_MSRVTT} \
#     --output_dir ${SAVE_DIR}/MSRVTT/LongVA \
#     --output_name ${CHUNKS}_${IDX} \
#     --llama3_path ${LLAMA_3} \
#     --num_chunks $CHUNKS \
#     --chunk_idx $IDX &
# done

# wait

# output_file=${SAVE_DIR}/MSRVTT/LongVA/merge.jsonl

# # Clear out the output file if it exists.
# > "$output_file"

# # Loop through the indices and concatenate each file.
# for IDX in $(seq 0 $((CHUNKS-1))); do
#     cat ${SAVE_DIR}/MSRVTT/LongVA/${CHUNKS}_${IDX}.json >> "$output_file"
# done

# python /13390024681/llama/EfficientVideo/FreeVA/scripts/gpt_eval/calculate_score.py \
#     --output_dir ${SAVE_DIR}/MSRVTT/LongVA \
#     --output_name merge
