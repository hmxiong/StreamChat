import os
import json
import math
import torch
from tqdm import tqdm
from PIL import Image
from transformers import AutoModel, AutoTokenizer
from decord import VideoReader, cpu    # pip install decord
import argparse

MAX_NUM_FRAMES=8 # if cuda OOM set a smaller number

def encode_video(video_path):
    def uniform_sample(l, n):
        gap = len(l) / n
        idxs = [int(i * gap + gap / 2) for i in range(n)]
        return [l[i] for i in idxs]

    vr = VideoReader(video_path, ctx=cpu(0))
    sample_fps = round(vr.get_avg_fps() / 1)  # FPS
    frame_idx = [i for i in range(0, len(vr), sample_fps)]
    if len(frame_idx) > MAX_NUM_FRAMES:
        frame_idx = uniform_sample(frame_idx, MAX_NUM_FRAMES)
    frames = vr.get_batch(frame_idx).asnumpy()
    frames = [Image.fromarray(v.astype('uint8')) for v in frames]
    print('num frames:', len(frames))
    return frames

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
    print("init model from {}".format(args.model_path))
    model = AutoModel.from_pretrained(args.model_path, trust_remote_code=True,
        attn_implementation='sdpa', torch_dtype=torch.bfloat16) # sdpa or flash_attention_2, no eager
    model = model.eval().cuda()
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)

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
    for sample in tqdm(gt_questions, desc="MiniCMP_v2.6 Inference for StreamBench"):
        video_name = sample['info']['video_path']
        class_1 = sample['info']['class_1']
        class_2 = sample['info']['class_2']
        
        temp_path = os.path.join(video_dir, class_1, f"{video_name}")
        if os.path.exists(temp_path):
            video_path = temp_path
    
            frames = encode_video(video_path)
        
        for ques_sample in sample['breakpoint']:
            
            question = ques_sample['question']
            answer = ques_sample['answer']
            id = ques_sample['time']
            qa_class = ques_sample['class']
            
            index += 1

            sample_set = {'id': id, 'question': question, 'answer': answer, 'class': qa_class}

            # Load the video file
            # for fmt in video_formats:  # Added this line
            # Set decode params for video
            msgs = [
                {'role': 'user', 'content': frames + [question]}, 
            ]
            
            params={}
            params["use_image_id"] = False
            params["max_slice_nums"] = 2 # use 1 if cuda OOM and video resolution >  448*448

            pred = model.chat(
                image=None,
                msgs=msgs,
                tokenizer=tokenizer,
                **params
            )
            
            sample_set['pred'] = pred
            output_list.append(sample_set)
            ans_file.write(json.dumps(sample_set) + "\n")
            
    ans_file.close()

if __name__ == "__main__":
    args = parse_args()
    run_inference(args)