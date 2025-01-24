import math
import os
import argparse
import json

from tqdm import tqdm
from llava.eval.model_utils import load_video

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path

from PIL import Image
import math
import torch
import time

def llava_inference(video_frames, question, conv_mode, model, tokenizer, image_processor, image_sizes):
    if model.config.mm_use_im_start_end:
        qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + question
    else:
        qs = DEFAULT_IMAGE_TOKEN + '\n' + question

    conv = conv_templates[conv_mode].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()
    # print(question)
    input_ids, ques_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt', question=question)
    input_ids = input_ids.unsqueeze(0).cuda()
    ques_ids = ques_ids.unsqueeze(0).cuda()
    # print("input",input_ids)
    # print("ques",ques_ids)
    
    image_tensor = process_images(video_frames, image_processor, model.config)  

    with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                images=image_tensor.to(dtype=torch.float16, device='cuda', non_blocking=True),
                image_sizes=image_sizes,
                question_ids=ques_ids,
                do_sample=True if args.temperature > 0 else False,
                temperature=args.temperature,
                top_p=args.top_p,
                num_beams=args.num_beams,
                max_new_tokens=128,
                use_cache=True)

    outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
    return outputs

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
    parser.add_argument('--video_dir', help='Directory containing video files.', required=True)
    parser.add_argument('--gt_file_question', help='Path to the ground truth file containing question.', required=True)
    parser.add_argument('--gt_file_answers', help='Path to the ground truth file containing answers.', required=True)
    parser.add_argument('--output_dir', help='Directory to save the model results JSON.', required=True)
    parser.add_argument('--output_name', help='Name of the file for storing results JSON.', required=True)
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--conv-mode", type=str, required=False, default='video-chatgpt_v1')
    parser.add_argument("--num_chunks", type=int, default=1)
    parser.add_argument("--chunk_idx", type=int, default=0)
    parser.add_argument("--num_frames", type=int, default=100)
    parser.add_argument("--device", type=str, required=False, default='cuda:0')
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=None)

    return parser.parse_args()


def run_inference(args):
    """
    Run inference on Video QA DataSetå.

    Args:
        args: Command-line arguments.
    """

    disable_torch_init()
    model_path = os.path.expanduser(args.model_name)
    model_name = get_model_name_from_path(model_path)
    print("Initialize baseline 2 !")
    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, args.model_base, model_name)


    gt_questions = json.load(open(args.gt_file_question, "r"))
    # if
    gt_questions = get_chunk(gt_questions, args.num_chunks, args.chunk_idx)
    # gt_answers = json.load(open(args.gt_file_answers, "r"))
    # gt_answers = get_chunk(gt_answers, args.num_chunks, args.chunk_idx)

    answers_file = os.path.join(args.output_dir, f"{args.output_name}.json")
    os.makedirs(args.output_dir, exist_ok=True)
    ans_file = open(answers_file, "w")

    # Create the output directory if it doesn't exist
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    output_list = []  # List to store the output results
    conv_mode = args.conv_mode

    video_formats = ['.mp4', '.avi', '.mov', '.mkv']

    # Iterate over each sample in the ground truth file
    index = 0
    sample_num = 100
    
    video_dir = args.video_dir
    if "msvd" in video_dir:
        mode = "MSVD"
    elif "MSRVTT" in args.video_dir:
        mode = "MSRVTT"
    elif "ActiveNet" in args.video_dir:
        mode = "ActiveNet"
    else:
        mode = "Others"
    
    for sample in tqdm(gt_questions, desc="FreeVA Inference for:{}".format(mode)):
        # print(sample)
        # video_name = sample['video_name']
        # question = sample['question']
        # id = sample['question_id']
        video_name = sample['video']
        question = sample['question']
        answer = sample['answer']
        id = sample['question_id']
        # answer = gt_answers[index]['answer']
        index += 1

        sample_set = {'id': id, 'question': question, 'answer': answer}

        # Load the video file
        for fmt in video_formats:  # Added this line
            vid_name = f"v_{video_name}" if 'Activitynet' in args.video_dir else video_name
            temp_path = os.path.join(args.video_dir, f"{vid_name}")
                
            if os.path.exists(temp_path):
                # print(f'processing {idx}/{len(gt_questions)}')
                video_path = temp_path
                video_frames, sizes = load_video(video_path, num_frm=args.num_frames) # 实际上只抽取了num_frames对应的帧数                
                # Run inference on the video and add the output to the list
                output = llava_inference(video_frames, question, conv_mode, model,
                                                tokenizer, image_processor, sizes)
                # print(output)
                sample_set['pred'] = output
                output_list.append(sample_set)
                ans_file.write(json.dumps(sample_set) + "\n")
                break
        
        # if len(output_list) > sample_num:
        #     print("sample over !!")
        #     break
    ans_file.close()


if __name__ == "__main__":
    args = parse_args()
    run_inference(args)