CUDA_VISIBLE_DEVICES=0 python streaming_demo_2.py \
    --model_name "/13390024681/All_Model_Zoo/llava-v1.5-7b"  \
    --video_dir /13390024681/llama/EfficientVideo/Ours/6.mp4 \
    --conv-mode vicuna_v1 \
    --temperature 0.9 \
    --num_frames 4

CUDA_VISIBLE_DEVICES=0 python streaming_demo_0.py     --model_name "/13390024681/All_Model_Zoo/llava-v1.5-7b"      --video_dir /13390024681/llama/EfficientVideo/Ours/6.mp4     --conv-mode vicuna_v1     --temperature 0.9     --num_frames 2

CUDA_VISIBLE_DEVICES=0 python streaming_demo_llava_next.py     --model_name /13390024681/All_Model_Zoo/LLaVA-NExT-LLaMA3-8B      --video_dir /13390024681/llama/EfficientVideo/Ours/videos/6.mp4     --conv-mode llava_llama_3     --temperature 0.9     --num_frames 5 --mode on_line

CUDA_VISIBLE_DEVICES=0 python streaming_demo_llava_next_2.py     --model_name /13390024681/All_Model_Zoo/LLaVA-NExT-LLaMA3-8B      --video_dir /13390024681/llama/EfficientVideo/Ours/videos/6.mp4     --conv-mode llava_llama_3     --temperature 0.9     --num_frames 4 --mode on_line

CUDA_VISIBLE_DEVICES=0 python streaming_demo_llava_next_3.py     --model_name /13390024681/All_Model_Zoo/LLaVA-NExT-LLaMA3-8B      --video_dir /13390024681/llama/EfficientVideo/Ours/videos/6.mp4     --conv-mode llava_llama_3     --temperature 0.9     --num_frames 4 --mode on_line

CUDA_VISIBLE_DEVICES=0 python streaming_demo_llava_next_3.py  \
                    --model_name /13390024681/All_Model_Zoo/LLaVA-NExT-LLaMA3-8B \
                    --video_dir /13390024681/llama/EfficientVideo/Ours/videos/1.mp4 \
                    --conv-mode llava_llama_3  \
                    --temperature 0.2  \
                    --num_frames 4 \
                    --mode on_line \
                    --memory_basic_dir /13390024681/llama/EfficientVideo/Ours/memory_bank/memories \
                    --memory_file updata_memories_for_streaming.json \
                    --memory_search_top_k 1 \
                    --language en

CUDA_VISIBLE_DEVICES=0 python streaming_demo_longvlm_ego.py  \
                    --model_name /13390024681/All_Model_Zoo/LongVA-7B-DPO \
                    --video_dir /13390024681/llama/EfficientVideo/Ours/videos/ego_streaming/ego_1.mp4 \
                    --conv-mode qwen_1_5_ego  \
                    --temperature 0.2  \
                    --num_beams 1 \
                    --mode on_line \
                    --memory_basic_dir /13390024681/llama/EfficientVideo/Ours/memory_bank/memories \
                    --memory_file updata_memories_for_streaming.json \
                    --memory_search_top_k 1 \
                    --language en

CUDA_VISIBLE_DEVICES=0 python streaming_demo_longvlm_ego.py  \
                    --model_name /13390024681/All_Model_Zoo/LongVA-7B-DPO \
                    --video_dir /13390024681/All_Data/EgoSchema/EgoSampled/02344f2e-3d7a-423a-bcba-a5ef96cde81e.mp4 \
                    --conv-mode qwen_1_5_ego  \
                    --temperature 0.2  \
                    --num_beams 1 \
                    --mode on_line \
                    --memory_basic_dir /13390024681/llama/EfficientVideo/Ours/memory_bank/memories \
                    --memory_file updata_memories_for_streaming_test_02344f2e.json \
                    --save_file /13390024681/llama/EfficientVideo/Ours/output/result_test_02344f2e.json \
                    --memory_search_top_k 1 \
                    --language en \
                    --multi_modal_memory 

CUDA_VISIBLE_DEVICES=0 python streaming_demo_longvlm_ego.py  \
                    --model_name /13390024681/All_Model_Zoo/LongVA-7B-DPO \
                    --video_dir /13390024681/All_Data/Streaming_final/Cooking_show/SvLW2h8Ob_s.mp4 \
                    --conv-mode qwen_1_5_ego  \
                    --temperature 0.2  \
                    --num_beams 1 \
                    --mode on_line \
                    --memory_basic_dir /13390024681/llama/EfficientVideo/Ours/memory_bank/memories \
                    --memory_file updata_memories_for_streaming_test_SvLW2h8Ob_s.json \
                    --save_file /13390024681/llama/EfficientVideo/Ours/output/result_test_SvLW2h8Ob_s.json \
                    --memory_search_top_k 1 \
                    --language en \
                    --multi_modal_memory 

# fast and new inference model 实际测试的时候使用的
CUDA_VISIBLE_DEVICES=0,1 python streaming_demo_longvlm_ego_fast.py  \
                    --model_name /13390024681/All_Model_Zoo/LongVA-7B-DPO \
                    --video_dir /13390024681/All_Data/Streaming_final/Comedy_drama/WNymWAjIB_k.mp4 \
                    --conv-mode qwen_1_5_ego  \
                    --temperature 0.2  \
                    --num_beams 1 \
                    --mode on_line \
                    --memory_basic_dir /13390024681/llama/EfficientVideo/Ours/memory_bank/memories \
                    --memory_file updata_memories_for_streaming_test_WNymWAjIB_k.json \
                    --save_file /13390024681/llama/EfficientVideo/Ours/output/result_test_WNymWAjIB_k.json \
                    --memory_search_top_k 1 \
                    --language en \
                    --multi_modal_memory 
                    
# /13390024681/llama/EfficientVideo/Ours/videos/ego_streaming
# /13390024681/All_Data/EgoSchema/EgoSampled
# /13390024681/All_Data/Streaming_final

################## total infernec #####################
CUDA_VISIBLE_DEVICES=0 python inference_streaming_longva.py  \
                    --model_name /13390024681/All_Model_Zoo/LongVA-7B-DPO \
                    --video_dir /13390024681/All_Data/Streaming_final \
                    --annotations /13390024681/llama/EfficientVideo/Ours/annotations/my.json \
                    --conv-mode qwen_1_5_ego  \
                    --temperature 0.2  \
                    --num_beams 1 \
                    --mode on_line \
                    --memory_basic_dir /13390024681/llama/EfficientVideo/Ours/memory_bank/memories/my \
                    --memory_file updata_memories_for_streaming_test_my.json \
                    --save_file /13390024681/llama/EfficientVideo/Ours/output/result_test_my.json \
                    --memory_search_top_k 1 \
                    --language en \
                    --multi_modal_memory 

CUDA_VISIBLE_DEVICES=0 python inference_streaming_longva.py  \
                    --model_name /13390024681/All_Model_Zoo/LongVA-7B-DPO \
                    --video_dir /13390024681/All_Data/Supplement_1 \
                    --annotations /13390024681/llama/EfficientVideo/Ours/annotations/hm.json \
                    --conv-mode qwen_1_5_ego  \
                    --temperature 0.2  \
                    --num_beams 1 \
                    --mode on_line \
                    --memory_basic_dir /13390024681/llama/EfficientVideo/Ours/memory_bank/memories/hm \
                    --memory_file updata_memories_for_streaming_test_hm.json \
                    --save_file /13390024681/llama/EfficientVideo/Ours/output/result_test_hm.json \
                    --memory_search_top_k 1 \
                    --language en \
                    --multi_modal_memory 

################# fast totlal inference ###############
CUDA_VISIBLE_DEVICES=0,1 python inference_streaming_longva_fast.py  \
                    --model_name /13390024681/All_Model_Zoo/LongVA-7B-DPO \
                    --video_dir /13390024681/All_Data/Streaming_final/Cooking_show \
                    --annotations /13390024681/llama/EfficientVideo/Ours/streaming_bench.json \
                    --conv-mode qwen_1_5_ego  \
                    --temperature 0.2  \
                    --num_beams 1 \
                    --mode on_line \
                    --memory_basic_dir /13390024681/llama/EfficientVideo/Ours/memory_bank/memories \
                    --memory_file updata_memories_for_test_streaming_fast_v0.1.json \
                    --save_file /13390024681/llama/EfficientVideo/Ours/output/result_test_streaming_fast_v0.1.json \
                    --memory_search_top_k 1 \
                    --language en \
                    --multi_modal_memory 

################# 使用离线模式来进行模型的测试 ############### 需要进行记忆更新
CUDA_VISIBLE_DEVICES=0,1 python inference_streaming_longva_offline.py \
                    --model_name /13390024681/All_Model_Zoo/LongVA-7B-DPO \
                    --video_dir /13390024681/All_Data/Streaming_final/Cooking_show \
                    --annotations /13390024681/llama/EfficientVideo/Ours/streaming_bench.json \
                    --conv-mode qwen_1_5  \
                    --temperature 0.2  \
                    --sample_rate 0.12 \
                    --num_beams 1 \
                    --mode on_line \
                    --memory_basic_dir /13390024681/llama/EfficientVideo/Ours/memory_bank/memories/streamingbench \
                    --memory_file updata_memories_for_test_streaming_fast_v0.1.json \
                    --save_file /13390024681/llama/EfficientVideo/Ours/output/result_test_streaming_fast_v0.1_0.12rate.json \
                    --memory_search_top_k 1 \
                    --language en \
                    --multi_modal_memory 

################### 使用离线 + 采样模式来实现 #########################  不需要进行记忆更新
CUDA_VISIBLE_DEVICES=0 python inference_streaming_longva_sample.py \
                    --model_name /13390024681/All_Model_Zoo/LongVA-7B-DPO \
                    --video_dir /13390024681/All_Data/Streaming_final/Cooking_show \
                    --annotations /13390024681/llama/EfficientVideo/Ours/streaming_bench.json \
                    --conv-mode qwen_1_5  \
                    --temperature 0.2  \
                    --num_beams 1 \
                    --mode on_line \
                    --memory_basic_dir /13390024681/llama/EfficientVideo/Ours/memory_bank/memories/streamingbench_sample \
                    --memory_file updata_memories_for_test_streaming_fast_v0.1.json \
                    --save_file /13390024681/llama/EfficientVideo/Ours/output/result_test_streaming_fast_v0.1_sample.json \
                    --memory_search_top_k 1 \
                    --language en \
                    --multi_modal_memory 

################# class dataset #######################
CUDA_VISIBLE_DEVICES=0 python video_data_filter.py  \
                    --model_path /13390024681/All_Model_Zoo/LongVA-7B-DPO \
                    --video_dir /13390024681/All_Data \
                    --video_path_list /13390024681/All_Data/YouTube-8M/only_movie.json \
                    --output_dir /13390024681/llama/EfficientVideo/Ours/tools \
                    --output_name Youtube_movie

CUDA_VISIBLE_DEVICES=0 python video_data_filter.py  \
                    --model_path /13390024681/All_Model_Zoo/LongVA-7B-DPO \
                    --video_dir /13390024681/All_Data/EgoSchema/good_clips_git \
                    --video_path_list /13390024681/All_Data/data/egoschema/fullset_anno.json \
                    --output_dir /13390024681/llama/EfficientVideo/Ours/tools \
                    --output_name EgoSchema_filtered

################### eval with llama3 ######################
#!/usr/bin/env bash
source ~/anaconda3/etc/profile.d/conda.sh
conda activate freeva

LLAMA_3=/13390024681/All_Model_Zoo/llama3-8b-instruct-hf
SAVE_DIR=/13390024681/llama/EfficientVideo/All_Score

VIDEO_LLAVA_STREAMING=/13390024681/All_Data/Streaming_Zero_Shot_QA/Video-LLaVA-7B/merge.jsonl

LLAMA_VID_STREAMING=/13390024681/llama/EfficientVideo/LLaMA-VID/work_dirs/eval_streaming/llama-vid/llama-vid-7b-full-224-video-fps-1/merge.jsonl

# FREEVA_STREAMING=/13390024681/llama/EfficientVideo/FreeVA/output/MSVD_Zero_Shot_QA/llava-v1.5-7b_u4FRS/merge.jsonl

MOVIECHAT_STREAMING=/13390024681/llama/EfficientVideo/MovieChat/output/moviechat_streaming_test.json

CUDA_VISIBLE_DEVICES="0,1"
gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -ra GPULIST <<< "$gpu_list"
CHUNKS=${#GPULIST[@]}

yes | pip install json_lines

mkdir ${SAVE_DIR}/Streaming/MovieChat
CUDA_VISIBLE_DEVICES=0 python3 /13390024681/llama/EfficientVideo/Ours/eval_ego_streaming_with_llama3.py \
    --predict_file /13390024681/llama/EfficientVideo/Ours/output/Ego-Streaming/all_result.json \
    --output_dir /13390024681/llama/EfficientVideo/Ours/output/Ego-Streaming \
    --output_name Ours \
    --llama3_path /13390024681/All_Model_Zoo/llama3-8b-instruct-hf \
    --num_chunks 0 \
    --chunk_idx 0 


wait

output_file=${SAVE_DIR}/Streaming/MovieChat/merge.jsonl

# Clear out the output file if it exists.
> "$output_file"

# Loop through the indices and concatenate each file.
for IDX in $(seq 0 $((CHUNKS-1))); do
    cat ${SAVE_DIR}/Streaming/MovieChat/${CHUNKS}_${IDX}.json >> "$output_file"
done

python /13390024681/llama/EfficientVideo/FreeVA/scripts/gpt_eval/calculate_score.py \
    --output_dir /13390024681/llama/EfficientVideo/Ours/output/Ego-Streaming \
    --output_name Ours