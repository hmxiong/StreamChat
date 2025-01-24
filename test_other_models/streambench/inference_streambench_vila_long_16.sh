CKPT_NAME="LongVILA_8B_16"
model_path="/13390024681/All_Model_Zoo/Llama-3-LongVILA-8B-1024Frames"
# GPT_Zero_Shot_QA="/13390024681/All_Data"
video_dir="/13390024681/All_Data/Streaming_Bench_v0.3"
gt_file_question="/13390024681/llama/EfficientVideo/Ours/streaming_bench_v0.3.json"
# gt_file_answers="${GPT_Zero_Shot_QA}/msvd/annotations/test_a.json"
output_dir="/13390024681/llama/EfficientVideo/Ours/output/StreamingBench_v0.3/${CKPT_NAME}"

pip install datasets

cd /13390024681/llama/EfficientVideo/scaling_on_scales
python setup.py install
cd /13390024681/llama/EfficientVideo/Ours

CUDA_VISIBLE_DEVICES="0,1"
gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -ra GPULIST <<< "$gpu_list"

CHUNKS=${#GPULIST[@]}


for IDX in $(seq 0 $((CHUNKS-1))); do
  CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python3 /13390024681/llama/EfficientVideo/Ours/test_other_models/streambench/inference_streambench_vila_long.py \
      --model_path ${model_path} \
      --video_dir ${video_dir} \
      --conv-mode llama_3 \
      --num-video-frames 16 \
      --gt_file_question ${gt_file_question} \
      --output_dir ${output_dir} \
      --output_name ${CHUNKS}_${IDX} \
      --num_chunks $CHUNKS \
      --chunk_idx $IDX &
done

wait

output_file=${output_dir}/streamingbench_merge.jsonl

# Clear out the output file if it exists.
> "$output_file"

# Loop through the indices and concatenate each file.
for IDX in $(seq 0 $((CHUNKS-1))); do
    cat ${output_dir}/${CHUNKS}_${IDX}.json >> "$output_file"
done
