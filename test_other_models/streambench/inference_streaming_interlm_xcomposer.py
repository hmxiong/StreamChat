import os
import numpy as np
import torch
import argparse
import math
import json
import torchvision.transforms as T
from tqdm import tqdm
from decord import VideoReader, cpu
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel, AutoTokenizer
import shutil
 
def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]

def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]

def parse_args():
    """
    Parse command-line arguments.
    """
    parser = argparse.ArgumentParser()

    # Define the command-line arguments
    parser.add_argument('--model_path', help='', required=True)
    parser.add_argument('--video_dir', help='Directory containing video files.', required=True)
    parser.add_argument('--gt_file_question', help='Path to the ground truth file containing question.', required=True)
    parser.add_argument('--output_dir', help='Directory to save the model results JSON.', required=True)
    parser.add_argument('--output_name', help='Name of the file for storing results JSON.', required=True)
    parser.add_argument("--num_chunks", type=int, default=1)
    parser.add_argument("--chunk_idx", type=int, default=0)
    parser.add_argument("--device", type=str, required=False, default='cuda:0')
    parser.add_argument('--model_base', help='', default=None, type=str, required=False)
    parser.add_argument("--model_max_length", type=int, required=False, default=2048)

    return parser.parse_args()

def run_inference(args):
    # If you have an 80G A100 GPU, you can put the entire model on a single GPU.
    # Otherwise, you need to load a model using multiple GPUs, please refer to the `Multiple GPUs` section.
    # print("copy file from another direction ")
    
    # shutil.copy("/13390024681/llama/EfficientVideo/Ours/ixc_utils.py", "/root/.cache/huggingface/modules/transformers_modules/internlm-xcomposer2d5-7b/ixc_utils.py")
    torch.set_grad_enabled(False)
    print("init model from {}".format(args.model_path))
    path = args.model_path
    model = AutoModel.from_pretrained(
        path,
        torch_dtype=torch.bfloat16,
        # device_map='cuda',
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        local_files_only=True
        ).eval().cuda()
    tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True, use_fast=False)
    model.tokenizer = tokenizer
    print(model.device, model.dtype)
    
    # prepare for generation
    gt_questions = json.load(open(args.gt_file_question, "r"))
    gt_questions = get_chunk(gt_questions, args.num_chunks, args.chunk_idx)

    answers_file = os.path.join(args.output_dir, f"{args.output_name}.json")
    os.makedirs(args.output_dir, exist_ok=True)
    ans_file = open(answers_file, "w")

    # Create the output directory if it doesn't exist
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    output_list = []  # List to store the output results

    video_dir = args.video_dir

    # Iterate over each sample in the ground truth file
    index = 0
    for sample in tqdm(gt_questions, desc="InterLM_XCP Inference for StreamBench"):
        video_name = sample['info']['video_path']
        class_1 = sample['info']['class_1']
        class_2 = sample['info']['class_2']
        
        # Load the video file
        # for fmt in video_formats:  # Added this line
        temp_path = os.path.join(video_dir, class_1, f"{video_name}")
        if os.path.exists(temp_path):
            video_path = temp_path
            image = [video_path,]
        
        for ques_sample in sample['breakpoint']:
            
            question = ques_sample['question']
            answer = ques_sample['answer']
            id = ques_sample['time']
            qa_class = ques_sample['class']
            
            index += 1

            sample_set = {'id': id, 'question': question, 'answer': answer, 'class': qa_class}
                        
            # with torch.autocast():
            # device_type='cuda', dtype=torch.float16
            with torch.no_grad():
                response, _ = model.chat(tokenizer, question, image, do_sample=False, use_meta=True)
            # print(response)
            # print(f'User: {question}\nAssistant: {response}')
            sample_set['pred'] = response
                
            output_list.append(sample_set)
            ans_file.write(json.dumps(sample_set) + "\n")
            
    ans_file.close()


if __name__ == "__main__":
    args = parse_args()
    run_inference(args)