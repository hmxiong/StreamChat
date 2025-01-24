#!/usr/bin/env bash
source ~/anaconda3/etc/profile.d/conda.sh
conda activate freeva
cd /13390024681/llama/EfficientVideo/Ours

CUDA_VISIBLE_DEVICES=0,1 bash /13390024681/llama/EfficientVideo/Ours/test_other_models/inference_ego_streaming_freeva.sh

CUDA_VISIBLE_DEVICES=0,1 bash /13390024681/llama/EfficientVideo/Ours/test_other_models/inference_ego_streaming_llavanext.sh

CUDA_VISIBLE_DEVICES=0,1 bash /13390024681/llama/EfficientVideo/Ours/test_other_models/inference_ego_streaming_longva.sh 8

CUDA_VISIBLE_DEVICES=0,1 bash /13390024681/llama/EfficientVideo/Ours/test_other_models/inference_ego_streaming_longva.sh 16

CUDA_VISIBLE_DEVICES=0,1 bash /13390024681/llama/EfficientVideo/Ours/test_other_models/inference_ego_streaming_longva.sh 32

CUDA_VISIBLE_DEVICES=0,1 bash /13390024681/llama/EfficientVideo/Ours/test_other_models/inference_qa_streaming_llava_hound.sh

CUDA_VISIBLE_DEVICES=0,1 bash /13390024681/llama/EfficientVideo/Video-LLaVA/run_qa_streaming.sh

CUDA_VISIBLE_DEVICES=0,1 bash /13390024681/llama/EfficientVideo/LLaMA-VID/streaming_eval.sh

CUDA_VISIBLE_DEVICES=0,1 bash /13390024681/llama/EfficientVideo/Flash-VStream/inference_ego_streaming_flash.sh

bash /13390024681/llama/EfficientVideo/MovieChat/run_test_ego_streaming.sh

CUDA_VISIBLE_DEVICES=0 bash /13390024681/llama/EfficientVideo/eval_ego_streaming.sh