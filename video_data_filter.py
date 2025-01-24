import os
import cv2
import json
import time
import numpy as np
import math
import argparse
import re
import ast
import torch
import transformers
from tqdm import tqdm
from longva.conversation import conv_templates, SeparatorStyle
from longva.constants import DEFAULT_IM_START_TOKEN, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_END_TOKEN, IMAGE_TOKEN_INDEX
from longva.mm_utils import get_model_name_from_path, tokenizer_image_token, KeywordsStoppingCriteria
from longva.model.builder import load_pretrained_model
from decord import VideoReader, cpu
# from longva.model.language_model.llava_llama import LlavaLlamaForCausalLM
from llava.eval.model_utils import load_video
# from ..utiles import Optical_flow
# from transformers import CLIPVisionModel, CLIPImageProcessor, CLIPVisionConfig

# image_processor = CLIPImageProcessor.from_pretrained(vision_tower_name)

ego_file_path = "/13390024681/All_Data/data/egoschema/fullset_anno.json"
ego_video_path = "/13390024681/All_Data/EgoSchema/good_clips_git"

video_path = "/13390024681/All_Data/YouTube-8M"
video_file_path = "/13390024681/All_Data/YouTube-8M/files_structure.json"
    
def parse_args():
    """
    Parse command-line arguments.
    """
    parser = argparse.ArgumentParser()

    # Define the command-line arguments
    parser.add_argument('--model_path', help='The path to save the model', default="/13390024681/All_Model_Zoo/LongVA-7B-DPO", required=False)
    parser.add_argument('--video_dir', help='Directory containing video files.', default=video_path , required=False)
    parser.add_argument('--video_path_list', help='Path to the file store the video path list', default=video_file_path, required=False)
    parser.add_argument('--output_dir', help='Directory to save the model results JSON.', default="/13390024681/llama/EfficientVideo/Ours/tools", required=False)
    parser.add_argument('--output_name', help='Name of the file for storing results JSON.', default="Youtube_filtered", required=False)
    parser.add_argument("--num_chunks", type=int, default=1)
    parser.add_argument("--chunk_idx", type=int, default=0)
    parser.add_argument("--device", type=str, required=False, default='cuda:0')
    parser.add_argument('--model_base', help='', default=None, type=str, required=False)
    parser.add_argument("--model_max_length", type=int, required=False, default=2048)
    
    return parser.parse_args()
    
def check_file_exist(file_paths):
    all_data = len(file_paths)
    miss_file = []
    count = 0
    for file_path in tqdm.tqdm(file_paths, desc="Checking dataset..."):
        if os.path.exists(file_path):
            count += 1
        else:
            miss_file.append(file_path)
    if count == all_data:
        print("All Data Ready !!")
    else:
        with open("miss_data.json", "w") as f:
            json.dump(miss_file, f, indent=4)
            
        print("Missing data :{}/{}".format(count, all_data))
        

def main(file_path, video_path):
    video_path_list = []
    if "ego" in file_path:
        with open(file_path, "r") as f:
            anno_data = json.load(f)
        
        for anno in tqdm.tqdm(anno_data, desc="Filtering dataset..."):
            # print(anno)
            video_name = anno_data[anno]['q_uid']+'.mp4'
            video_path_list.append(os.path.join(video_path, video_name))
        
        check_file_exist(video_path_list)
    elif "YouTube" in file_path:
        with open(file_path, "r") as f:
            anno_data = json.load(f)
        
        for class_name in tqdm.tqdm(anno_data.keys(), desc="Filtering dataset..."):
            for video_name in anno_data[class_name]:
                video_path_list.append(os.path.join(video_path, class_name, video_name))
        
        check_file_exist(video_path_list)
        
        filter_video_with_OpticalFlow(video_path_list, 2.0, "/13390024681/llama/EfficientVideo/Ours/tools/YouTube.json")

def calculate_optical_flow(video_path, threshold):
    cap = cv2.VideoCapture(video_path)
    
    # 检查视频是否打开成功
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return

    ret, prev_frame = cap.read()
    if not ret:
        print(f"Error: Could not read the first frame of video {video_path}")
        return
    
    # 转换为灰度图
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    
    count_exceeding_threshold = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # 计算光流
        flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        
        # 计算光流的幅度
        magnitude, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        
        # 记录幅度超过阈值的次数
        if np.any(magnitude > threshold):
            count_exceeding_threshold += 1
        
        prev_gray = gray
    
    cap.release()
    print(f"Number of frames exceeding the threshold: {count_exceeding_threshold}")
    
def filter_video_with_OpticalFlow(video_path_list,  threshold, save_path):
    count_list = []
    pbar = tqdm.tqdm(total=len(video_path_list), desc="Filter video with Optical-FLow ....", unit="video")
    for video_path in video_path_list:
        time_1 = time.time()
        cap = cv2.VideoCapture(video_path)
        # 检查视频是否打开成功
        if not cap.isOpened():
            print(f"Error: Could not open video {video_path}")
            return
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        frame_interval = fps 
        ret, prev_frame = cap.read()
        frame_count = 0
        while frame_count % frame_interval != 0:
            ret, prev_frame = cap.read()
            frame_count += 1
        if not ret:
            print(f"Error: Could not read the first frame of video {video_path}")
            return
        
        # 转换为灰度图
        # 将第一帧图像调整为224x224像素
        prev_frame = cv2.resize(prev_frame, (112, 112))
        
        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        
        count_exceeding_threshold = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_count % frame_interval != 0:
                continue
            
            frame = cv2.resize(frame, (112, 112))
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # 计算光流
            flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            
            # 计算光流的幅度
            magnitude, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
            
            # 记录幅度超过阈值的次数
            if np.any(magnitude > threshold):
                count_exceeding_threshold += 1
            
            prev_gray = gray
        time_2 = time.time()
        pbar.set_postfix(time="{:.2f}".format(time_2-time_1))
        pbar.update(1)
        cap.release()
        count_list.append({"file_path":video_path, "count":count_exceeding_threshold})
        print(f"Number of frames exceeding the threshold: {count_exceeding_threshold}")
    
    with open(save_path, "w") as f:
        json.dump(count_list, f)
    print("Saving result in :{}".format(save_path))

def get_model_output(model, video_processor, tokenizer, video, qs, args):
    if model.config.mm_use_im_start_end:
        qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
    else:
        qs = DEFAULT_IMAGE_TOKEN + '\n' + qs

    # conv_mode = "qwen_1_5"
    # args.conv_mode = conv_mode

    conv = conv_templates["qwen_1_5"].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()


    # video_tensor = video_processor.preprocess(video, return_tensors='pt')['pixel_values'][0].half().to(args.device)
    video_tensor = video_processor.preprocess(video, return_tensors="pt")["pixel_values"].to(model.device, dtype=torch.float16)
    # video_tensor = torch.cat([video_tensor[:len(video_tensor)//2], video_processor, video_tensor[len(video_tensor)//2:]], dim=0)
    # print(video_tensor.shape)
    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(args.device)

    # stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
    # keywords = [stop_str]
    # stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)

    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            images=[video_tensor],
            modalities=["video"],
            do_sample=True,
            temperature=0.1,
            max_new_tokens=25,
            num_beams=1,
            use_cache=False)

    outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
    # input_token_len = input_ids.shape[1]
    # n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
    # if n_diff_input_output > 0:
    #     print(f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')
    # outputs = tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0]
    # outputs = outputs.strip()
    # if outputs.endswith(stop_str):
    #     outputs = outputs[:-len(stop_str)]
    # outputs = outputs.strip()
    print(outputs)
    return outputs

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

def run_inference(args):
    """
    Run inference on ActivityNet QA DataSet using the Video-ChatGPT model.

    Args:
        args: Command-line arguments.
    """
    # Initialize the model
    print("Initialize LongVA !")
    model_name = get_model_name_from_path(args.model_path)
    tokenizer, model, processor, context_len = load_pretrained_model(args.model_path, None , "llava_qwen")
    model = model.to(args.device)


    video_path_list = json.load(open(args.video_path_list, "r"))
    # video_path_list = get_chunk(video_path_list, args.num_chunks, args.chunk_idx)
    
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
    elif "ActiveNet" in args.video_dir:
        mode = "ActiveNet"
    else:
        mode = "Others"
        
    video_formats = ['.mp4', '.avi', '.mov', '.mkv']

    target_class= 'Movieclips'
    class_list = {"A":"Drama", "B":"Action", "C":"Cartoon", "D":"Romance", "E":"Sci-fi", "F":"Others"}
    # Iterate over each sample in the ground truth file
    index = 0
    for sample in tqdm(video_path_list[target_class], desc="LongVA Inference for:{}".format(mode)):
        video_name = sample
        
        video_path = os.path.join(args.video_dir, target_class, video_name)
        index += 1

        sample_set = {'name': video_name, 'video_class':target_class}

        question = "Based on the video information you see, please help categorize the video into one of the following categories: {A.Drama, B.Action, C.Cartoon, D.Romance, E.Sci-fi, F.Others}.\
                    Please generate the response in the form of a Python dictionary string with keys 'pred' and 'score', where value of 'pred' is the uppercase STRING corresponding to the category option. \
                    DO NOT PROVIDE ANY OTHER OUTPUT TEXT OR EXPLANATION. Only provide the Python dictionary string. \
                    For example, your response should look like this: {'pred': 'A'}."
        # Load the video file

        # try:
        # Run inference on the video and add the output to the list
        video_frame, video_size = load_video(video_path, num_frm=64)
        output = get_model_output(model, processor, tokenizer, video_frame, question, args)
        print(output)
        matches = re.findall(r'\{.*?\}', output)
        for match in matches:
            result = ast.literal_eval(match)
        
        sample_set['pred'] = result['pred']
        sample_set['category'] = class_list[sample_set['pred']]
        print(result['pred'], class_list[sample_set['pred']])
        output_list.append(sample_set)
        # except Exception as e:
        #     print(f"Error processing video file '{video_name}': {e}")
        # ans_file.write(json.dumps(sample_set) + "\n")
        # break

    # ans_file.close()
    # Save the output list to a JSON file
    with open(os.path.join(args.output_dir, f"{args.output_name}.json"), 'w') as file:
        json.dump(output_list, file, indent=4)
    
def run_inference_ego(args):
    """
    Run inference on ActivityNet QA DataSet using the Video-ChatGPT model.

    Args:
        args: Command-line arguments.
    """
    new_video_path_list = []
    video_path_list = json.load(open(args.video_path_list, "r"))
    for sample_name in tqdm(video_path_list.keys(), desc="prepare data for chunk"):
        new_video_path_list.append(video_path_list[sample_name])
        
    video_path_list = get_chunk(new_video_path_list, args.num_chunks, args.chunk_idx)
    
    # Initialize the model
    print("Initialize LongVA !")
    model_name = get_model_name_from_path(args.model_path)
    tokenizer, model, processor, context_len = load_pretrained_model(args.model_path, None , "llava_qwen")
    model = model.to(args.device)
    
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
    elif "ActiveNet" in args.video_dir:
        mode = "ActiveNet"
    else:
        mode = "Others"
        
    video_formats = ['.mp4', '.avi', '.mov', '.mkv']

    # target_class= 'Movieclips'
    class_list = {"A":"Cooking", "B":"Construction", "C":"Room-Tour", "D":"Gardening", "E":"Others"}
    # Cooking、Construction、Room-Tour、Gardening、Others
    # Iterate over each sample in the ground truth file
    index = 0
    for sample in tqdm(video_path_list, desc="LongVA Inference for:{}".format(mode)):
        video_name = sample['q_uid'] +'.mp4'
        
        video_path = os.path.join(args.video_dir, video_name)
        index += 1

        sample_set = {'name': video_name}

        question = "Frederic Lane noted that Venice lost ships in battle at the end of the sixteenth century, showing that Venetian shipbuilding was no longer known for its reliability.\
                    Please generate the response in the form of a Python dictionary string with keys 'pred' and 'score', where value of 'pred' is the uppercase STRING corresponding to the category option. \
                    DO NOT PROVIDE ANY OTHER OUTPUT TEXT OR EXPLANATION. Only provide the Python dictionary string. \
                    For example, your response should look like this: {'pred': 'A'}."
        # Load the video file

        # try:
        # Run inference on the video and add the output to the list
        video_frame, video_size = load_video(video_path, num_frm=128)
        output = get_model_output(model, processor, tokenizer, video_frame, question, args)
        # print(output)
        matches = re.findall(r'\{.*?\}', output)
        for match in matches:
            result = ast.literal_eval(match)
        
        sample_set['pred'] = result['pred']
        sample_set['category'] = class_list[sample_set['pred']]
        # print(result['pred'], class_list[sample_set['pred']])
        output_list.append(sample_set)
        # except Exception as e:
        #     print(f"Error processing video file '{video_name}': {e}")
        ans_file.write(json.dumps(sample_set) + "\n")
        # break

    # ans_file.close()
    # Save the output list to a JSON file
    # with open(os.path.join(args.output_dir, f"{args.output_name}.json"), 'w') as file:
    #     json.dump(output_list, file, indent=4)

if __name__ == "__main__":
    # ego_file_path = "/13390024681/All_Data/data/egoschema/fullset_anno.json"
    # ego_video_path = "/13390024681/All_Data/EgoSchema/good_clips_git"
    
    # video_path = "/13390024681/All_Data/YouTube-8M"
    # video_file_path = "/13390024681/All_Data/YouTube-8M/files_structure.json"
    
    # main(ego_file_path, ego_video_path)
    # main(video_file_path, video_path)
    
    args = parse_args()
    if "Ego" in args.video_dir:
        print("Filtering Ego Data")
    # run_inference(args)
        run_inference_ego(args)
    else:
        print("Filtering Youtube Data")
        run_inference(args)