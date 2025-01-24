# This file is modified from https://github.com/haotian-liu/LLaVA/

import argparse
import os
import os.path as osp
import re
from io import BytesIO
import sys
sys.path.append("/13390024681/llama/EfficientVideo/VILA")

import requests
import torch
import json
import math
from PIL import Image
from tqdm import tqdm

from llava.constants import (
    DEFAULT_IM_END_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IMAGE_TOKEN,
    IMAGE_PLACEHOLDER,
    IMAGE_TOKEN_INDEX,
)
from llava.conversation import SeparatorStyle, conv_templates
from llava.mm_utils import KeywordsStoppingCriteria, get_model_name_from_path, process_images, tokenizer_image_token
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import opencv_extract_frames

def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]

def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]

def image_parser(args):
    out = args.image_file.split(args.sep)
    return out


def load_image(image_file):
    if image_file.startswith("http") or image_file.startswith("https"):
        print("downloading image from url", args.video_file)
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert("RGB")
    else:
        image = Image.open(image_file).convert("RGB")
    return image


def load_images(image_files):
    out = []
    for image_file in image_files:
        image = load_image(image_file)
        out.append(image)
    return out


def eval_model(args):
    # Model
    disable_torch_init()
    print("Init VILA model from local disk !!")
    model_name = get_model_name_from_path(args.model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(args.model_path, model_name, args.model_base)
    print("model loaded finished !!")
    
    if "llama-2" in model_name.lower():
        conv_mode = "llava_llama_2"
    elif "v1" in model_name.lower():
        conv_mode = "llava_v1"
    elif "mpt" in model_name.lower():
        conv_mode = "mpt"
    else:
        conv_mode = "llava_v0"

    if args.conv_mode is not None and conv_mode != args.conv_mode:
        print(
            "[WARNING] the auto inferred conversation mode is {}, while `--conv-mode` is {}, using {}".format(
                conv_mode, args.conv_mode, args.conv_mode
            )
        )
    else:
        args.conv_mode = conv_mode
    
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
    for sample in tqdm(gt_questions, desc="VILA Inference for StreamBench"):
        video_name = sample['info']['video_path']
        class_1 = sample['info']['class_1']
        class_2 = sample['info']['class_2']
        
        # Load the video file
        # for fmt in video_formats:  # Added this line
        temp_path = os.path.join(video_dir, class_1, f"{video_name}")
        if os.path.exists(temp_path):
            # video_path = temp_path
            # image = [video_path,]
            images, num_frames = opencv_extract_frames(temp_path, args.num_video_frames)
        
        for ques_sample in sample['breakpoint']:
            
            question = ques_sample['question']
            answer = ques_sample['answer']
            id = ques_sample['time']
            qa_class = ques_sample['class']
            
            index += 1

            sample_set = {'id': id, 'question': question, 'answer': answer, 'class': qa_class}
            
            qs = question
            image_token_se = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN
            if IMAGE_PLACEHOLDER in qs:
                if model.config.mm_use_im_start_end:
                    qs = re.sub(IMAGE_PLACEHOLDER, image_token_se, qs)
                else:
                    qs = re.sub(IMAGE_PLACEHOLDER, DEFAULT_IMAGE_TOKEN, qs)
            else:
                if DEFAULT_IMAGE_TOKEN not in qs:
                    print("no <image> tag found in input. Automatically append one at the beginning of text.")
                    # do not repeatively append the prompt.
                    if model.config.mm_use_im_start_end:
                        qs = (image_token_se + "\n") * len(images) + qs
                    else:
                        qs = (DEFAULT_IMAGE_TOKEN + "\n") * len(images) + qs
            # print("input: ", qs)

            conv = conv_templates[args.conv_mode].copy()
            conv.append_message(conv.roles[0], qs)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()

            images_tensor = process_images(images, image_processor, model.config).to(model.device, dtype=torch.float16)
            input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).cuda()

            stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
            keywords = [stop_str]
            stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)

            # print(images_tensor.shape)
            with torch.inference_mode():
                output_ids = model.generate(
                    input_ids,
                    images=[
                        images_tensor,
                    ],
                    do_sample=True if args.temperature > 0 else False,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    num_beams=args.num_beams,
                    max_new_tokens=args.max_new_tokens,
                    use_cache=True,
                    stopping_criteria=[stopping_criteria],
                )

            outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]
            outputs = outputs.strip()
            if outputs.endswith(stop_str):
                outputs = outputs[: -len(stop_str)]
            outputs = outputs.strip()
            # print("out_put",outputs)
            
            sample_set['pred'] = outputs
                
            output_list.append(sample_set)
            ans_file.write(json.dumps(sample_set) + "\n")
            
    ans_file.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # parser.add_argument("--model-path", type=str, default="Efficient-Large-Model/VILA-2.7b")
    # parser.add_argument("--model-base", type=str, default=None)
    # parser.add_argument("--image-file", type=str, default=None)
    # parser.add_argument("--video-file", type=str, default=None)
    # parser.add_argument("--num-video-frames", type=int, default=6)
    # parser.add_argument("--query", type=str, required=True)
    # parser.add_argument("--conv-mode", type=str, default=None)
    # parser.add_argument("--sep", type=str, default=",")
    parser.add_argument('--model_path', help='', required=True)
    parser.add_argument('--video_dir', help='Directory containing video files.', required=True)
    parser.add_argument('--gt_file_question', help='Path to the ground truth file containing question.', required=True)
    parser.add_argument('--output_dir', help='Directory to save the model results JSON.', required=True)
    parser.add_argument('--output_name', help='Name of the file for storing results JSON.', required=True)
    parser.add_argument("--conv-mode", type=str, default=None)
    parser.add_argument("--num_chunks", type=int, default=1)
    parser.add_argument("--chunk_idx", type=int, default=0)
    parser.add_argument("--device", type=str, required=False, default='cuda:0')
    parser.add_argument('--model_base', help='', default=None, type=str, required=False)
    parser.add_argument("--num-video-frames", type=int, default=6)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    args = parser.parse_args()

    eval_model(args)
