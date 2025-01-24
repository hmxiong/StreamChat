import math
import os
import argparse
import json
import sys
sys.path.append("/13390024681/llama/EfficientVideo/Ours")

import torch
import transformers
import numpy as np
from tqdm import tqdm
from llava_hound.constants import X_TOKEN_INDEX, X_INDEX_TOKEN, DEFAULT_X_TOKEN, DEFAULT_X_START_TOKEN, DEFAULT_X_END_TOKEN
from llava_hound.conversation import conv_templates, SeparatorStyle
from llava_hound.model.builder import load_pretrained_model
from llava_hound.utils import disable_torch_init
from llava_hound.mm_utils import tokenizer_X_token, process_images, get_model_name_from_path, KeywordsStoppingCriteria
from decord import VideoReader, cpu
# from longva.model.language_model.llava_llama import LlavaLlamaForCausalLM
from llava.eval.model_utils import load_video

# from longva.train.train import smart_tokenizer_and_embedding_resize


def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]

def load_video_longva(video_path, haystack_frames):
    vr = VideoReader(video_path, ctx=cpu(0))
    total_frame_num = len(vr)
    uniform_sampled_frames = np.linspace(0, total_frame_num - 1, haystack_frames, dtype=int)
    frame_idx = uniform_sampled_frames.tolist()
    frames = vr.get_batch(frame_idx).asnumpy()
    return frames
    
def parse_args():
    """
    Parse command-line arguments.
    """
    parser = argparse.ArgumentParser()

    # Define the command-line arguments
    parser.add_argument('--model_path', help='', required=True)
    parser.add_argument('--cache_dir', help='', required=True)
    parser.add_argument('--video_dir', help='Directory containing video files.', required=True)
    parser.add_argument('--gt_file_question', help='Path to the ground truth file containing question.', required=True)
    parser.add_argument('--gt_file_answers', help='Path to the ground truth file containing answers.', required=False)
    parser.add_argument('--output_dir', help='Directory to save the model results JSON.', required=True)
    parser.add_argument('--output_name', help='Name of the file for storing results JSON.', required=True)
    parser.add_argument("--num_chunks", type=int, default=1)
    parser.add_argument("--chunk_idx", type=int, default=0)
    parser.add_argument("--device", type=str, required=False, default='cuda:0')
    parser.add_argument('--model_base', help='', default=None, type=str, required=False)
    parser.add_argument("--model_max_length", type=int, required=False, default=2048)

    return parser.parse_args()

def llava_hound_inference(question, conv_mode, model, tokenizer, video_processor, video_path):
    if model.config.mm_use_x_start_end:
        # qs = X_INDEX_TOKEN[] + DEFAULT_X_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + question
        qs = DEFAULT_X_START_TOKEN['VIDEO'] + DEFAULT_X_TOKEN['VIDEO'] + DEFAULT_X_END_TOKEN['VIDEO'] + '\n' + question
    else:
        # qs = DEFAULT_IMAGE_TOKEN + '\n' + question
        qs = DEFAULT_X_TOKEN['VIDEO'] + '\n' + question

    conv = conv_templates[conv_mode].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()
    # print(question)
    # input_ids, ques_ids = tokenizer_X_token(prompt, tokenizer, X_TOKEN_INDEX['VIDEO'], return_tensors='pt', question=question)
    # input_ids = input_ids.unsqueeze(0).cuda()
    # ques_ids = ques_ids.unsqueeze(0).cuda()
    input_ids, ques_ids = tokenizer_X_token(prompt, tokenizer, X_TOKEN_INDEX['VIDEO'], return_tensors='pt', question=question)
    input_ids = input_ids.unsqueeze(0).to(args.device)
    ques_ids = ques_ids.unsqueeze(0).to(args.device)
    # print("input",input_ids)
    # print("ques",ques_ids)
    
    # image_tensor = process_images(video_frames, image_processor, model.config)  
    video_tensor = video_processor.preprocess(video_path, return_tensors='pt')['pixel_values'][0].half().to(args.device) # 经过搜索之后一共有10帧图片
    
    # print("video tensor:{}".format(video_tensor.shape)) # c T H W  其中T主要指的是8帧的视频, 每一帧图像的分辨率保持为 224 224 
    
    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
    keywords = [stop_str]
    stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)

    # if args.temperature < 0.01:
    #     args.temperature = -1 # greedy
    max_context_length = getattr(
        model.config, 'max_position_embeddings', 2048)
    max_new_tokens = min(max_context_length - input_ids.shape[1], 512)
    
    with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                images=[[video_tensor], ['video']],
                # image_sizes=image_sizes,
                # question_ids=ques_ids,
                do_sample=True,
                temperature=0.1,
                top_p=0.9,
                # num_beams=args.num_beams,
                max_new_tokens=max_new_tokens,
                use_cache=True,
                stopping_criteria=[stopping_criteria])

    # outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
    input_token_len = input_ids.shape[1]
    n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
    if n_diff_input_output > 0:
        print(f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')
    outputs = tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0]
    outputs = outputs.strip()
    if outputs.endswith(stop_str):
        outputs = outputs[:-len(stop_str)]
    outputs = outputs.strip()
    print(outputs)
    return outputs


def run_inference(args):
    """
    Run inference on ActivityNet QA DataSet using the Video-ChatGPT model.

    Args:
        args: Command-line arguments.
    """
    # Initialize the model
    print("Initialize LLaVA-Hound !")
    # model_name = get_model_name_from_path(args.model_path)
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    print("Model Name:{}".format(model_name))
    
    tokenizer, model, processor, context_len = load_pretrained_model(model_path, args.model_base, model_name)
    model = model.to(args.device)

    # Load both ground truth file containing questions and answers
    # with open(args.gt_file_question) as file:
    #     gt_questions = json.load(file)
    # with open(args.gt_file_answers) as file:
    #     gt_answers = json.load(file)

    gt_questions = json.load(open(args.gt_file_question, "r"))
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

    video_dir = args.video_dir
    
    if "msvd" in video_dir:
        mode = "MSVD"
    elif "MSRVTT" in args.video_dir:
        mode = "MSRVTT"
    elif "Act" in args.video_dir:
        mode = "ActiveNet"
    elif "NExT" in args.video_dir:
        mode = "NEXT"
    else:
        mode = "Others"
        
    video_formats = ['.mp4', '.webm', '.mkv']
    
    video_path_dict = {
        "/13390024681/All_Data/Streaming_final":["Cooking_show" ,"Comedy_drama" ,"Apple_TV"], 
        "/13390024681/All_Data/Supplement_1":["Cooking", "Metalworking"]
        }

    def find_key_by_category(category, data_dict):
        for key, categories in data_dict.items():
            if category in categories:
                return key
        return None
    
    conv_mode= "v1"
    # Iterate over each sample in the ground truth file
    index = 0
    for sample in tqdm(gt_questions, desc="LLaVA_Hound Inference for:{}".format(mode)):
        video_name = sample['video_name']
        question = sample['question']
        answer = sample['answer']
        id = sample['question_id']
        
        index += 1
        
        sample_set = {'id': id, 'question': question, 'answer': answer}

        # Load the video file
        for fmt in video_formats:  # Added this line
            temp_path = os.path.join(args.video_dir, f"v_{video_name}{fmt}")
            if os.path.exists(temp_path):
                video_path = temp_path
                # try:
                # Run inference on the video and add the output to the list
                output = llava_hound_inference(question, conv_mode, model,
                                                tokenizer, processor['video'], video_path)
                sample_set['pred'] = output
                output_list.append(sample_set)
                # except Exception as e:
                #     print(f"Error processing video file '{video_name}': {e}")
                ans_file.write(json.dumps(sample_set) + "\n")
                # break

    ans_file.close()
    # Save the output list to a JSON file
    # with open(os.path.join(args.output_dir, f"{args.output_name}.json"), 'w') as file:
    #     json.dump(output_list, file)


if __name__ == "__main__":
    args = parse_args()
    run_inference(args)
