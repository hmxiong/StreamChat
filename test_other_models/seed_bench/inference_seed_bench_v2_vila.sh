#!/usr/bin/env bash
source ~/anaconda3/etc/profile.d/conda.sh
conda activate freeva
cd /13390024681/llama/EfficientVideo/Ours

pip install datasets

cd /13390024681/llama/EfficientVideo/scaling_on_scales
python setup.py install
cd /13390024681/llama/EfficientVideo/Ours

CUDA_VISIBLE_DEVICES=0 python /13390024681/llama/EfficientVideo/Ours/test_other_models/seed_bench/inference_seed_bench_v2_vila.py