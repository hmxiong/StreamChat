#!/usr/bin/env bash
source ~/anaconda3/etc/profile.d/conda.sh
conda activate freeva
cd /13390024681/llama/EfficientVideo/Ours
# 218
yes | pip install json_lines

# BENCHMARK=Video_Bench
# TASK=Ours_rate0.2_chunk25
# OURS_EGO=/13390024681/llama/EfficientVideo/Ours/${BENCHMARK}/${TASK}.json
# LLAMA_3=/13390024681/All_Model_Zoo/llama3-8b-instruct-hf
# SAVE_DIR=/13390024681/llama/EfficientVideo/All_Score

CUDA_VISIBLE_DEVICES=0,1 python inference_video_bench.py \
                    --model_name /13390024681/All_Model_Zoo/LongVA-7B-DPO \
                    --conv-mode qwen_1_5  \
                    --temperature 0.2  \
                    --sample_rate 0.2 \
                    --chunk_size 25 \
                    --num_clusters 5 \
                    --interval 10 \
                    --short_window 20 \
                    --remember_window 5 \
                    --tau 5 \
                    --compress_rate 1 \
                    --num_beams 1 \
                    --mode on_line \
                    --memory_search_top_k 1 \
                    --language en