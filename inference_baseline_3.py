import math
import os
import argparse
import json

from tqdm import tqdm
# from llava.eval.model_utils import load_video

from llava_hound.constants import X_TOKEN_INDEX, X_INDEX_TOKEN, DEFAULT_X_TOKEN, DEFAULT_X_START_TOKEN, DEFAULT_X_END_TOKEN
from llava_hound.conversation import conv_templates, SeparatorStyle
from llava_hound.model.builder import load_pretrained_model
from llava_hound.utils import disable_torch_init
from llava_hound.mm_utils import tokenizer_X_token, process_images, get_model_name_from_path, KeywordsStoppingCriteria

from PIL import Image
import math
import torch
import time
import numpy as np
import cv2
from decord import VideoReader, cpu


def mean_absolute_error_torch(frame1, frame2, device='cuda'):
    frame1 = torch.tensor(frame1, device=device)
    frame2 = torch.tensor(frame2, device=device)
    
    return torch.sum(torch.abs(frame1 - frame2)).item()
    # return np.mean(np.abs(frame1 - frame2))
    
def mean_absolute_error(frame1, frame2):
    return np.mean(np.abs(frame1 - frame2))

def get_most_change_frame(frame_list, num_frm):
    """
    extract most change frame based on event changement
    """
    # 计算相邻帧之间的差异
    differences = []
    prev_frame = frame_list[0]
    # for i in range(1, len(frame_list)):
    for i in tqdm(range(0, len(frame_list)), desc="Calculating differences"):
        # diff = cv2.absdiff(frame_list[i], frame_list[i - 1])
        # diff_sum = np.sum(diff)
        current_frame = frame_list[i]
        diff = mean_absolute_error(prev_frame, current_frame)
        differences.append(diff)
        prev_frame = current_frame
    
    # num_frm+1个位置
    indices = np.argsort(differences)[-num_frm-1:]
    indices = sorted(indices)
    print(indices)
    # 从每段的中间抽取一帧
    segments = []
    prev_index = 0
    for index in indices:
        middle_frame_index = (prev_index + index) // 2
        segments.append(frame_list[middle_frame_index])
        prev_index = index + 1
    # 最后一段的中间帧
    middle_frame_index = (prev_index + len(frame_list) - 1) // 2
    segments.append(frame_list[middle_frame_index])
    
    return segments, differences

def load_video_with_event_chunk(video_path, num_frm):
    """
    Load video frames from a video file and segment frame based on frame comparasion.

    Parameters:
    vis_path (str): Path to the video file.
    num_frm (int): Number of frames to extract from each clip. Defaults to 100.

    Returns:
    list: List of PIL.Image.Image objects representing video frames.
    """
    # 创建一个视频捕获对象
    cap = cv2.VideoCapture(video_path)

    # 检查视频是否成功打开
    if not cap.isOpened():
        print("Error: Could not open video.")
        return []

    frames = []

    # 读取视频帧
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)

    # 释放视频捕获对象
    cap.release()
    
    important_frames = get_most_change_frame(frames, num_frm)

    return important_frames
    # print("time spend :{}".format(time_2 - time_1))
    # # decord.bridge.set_bridge('torch')
    # # Load video with VideoReader
    # vr = VideoReader(video_path, ctx=cpu(0))
    # total_frame_num = len(vr)

    # # Currently, this function supports only 1 clip
    
    # # Calculate total number of frames to extract
    # total_num_frm = min(total_frame_num, num_frm)
    # # Get indices of frames to extract
    # frame_idx = get_seq_frames(total_frame_num, total_num_frm)
    # # Extract frames as numpy array
    # img_array = vr.get_batch(frame_idx).asnumpy()   # T H W C

    # original_size = (img_array.shape[-2], img_array.shape[-3])  # (width, height)
    # original_sizes = (original_size,) * total_num_frm

    # clip_imgs = [Image.fromarray(img_array[j]) for j in range(total_num_frm)]
    

    # return clip_imgs, original_sizes

def read_images_in_video(video_path, num_frm):
    vr = VideoReader(video_path, ctx=cpu(0))
    total_frame_num = len(vr)
    fps = round(vr.get_avg_fps())
    frame_idx = [i for i in range(0, len(vr), fps)]
    spare_frames = vr.get_batch(frame_idx).asnumpy()
    original_size = (spare_frames.shape[-2], spare_frames.shape[-3])  # (width, height)
    original_sizes = (original_size,) * num_frm
    return spare_frames, frame_idx, original_sizes

# def semantic_search(video_tensor, question_ids, model, top_k):
    
#     question_embeddings  = model.get_model().embed_tokens(torch.tensor(question_ids, dtype=torch.long, device=video_tensor.device))
#     # print("question embeddings : {}".format(question_embeddings.shape)) # num_text_token 4096
#     # print("There are total :{} frames in this video.".format(total_frames)) # how to speed up ?

#     # all_image_features = torch.cat(feature_list)
#     C, T, H, W = video_tensor.shape
#     # all_image_features = video_tensor.permute(1, 0, 2, 3).reshape(T, C, -1)
#     video_features = model.encode_videos(video_tensor)
#     print("all_image_features shape :{}".format(all_image_features.shape))
#     # time_2 = time.time()
#     simarity = all_image_features @ question_embeddings.permute(1, 0) # num_frame 576 num_text_token
#     simarity = simarity.sum(dim=1).sum(dim=1)
#     # time_3 = time.time()
#     # print("simarity shape :{}".format(simarity.shape))
#     topk_values, topk_indices = torch.topk(simarity, num_frm)
#     topk_features = [feature_list[idx] for idx in topk_indices]
#     # Concatenate all top-k features
#     concatenated_features = torch.cat(topk_features, dim=0) # num_select 576 4096
#     concatenated_features = concatenated_features.view(-1, concatenated_features.shape[-1])
#     # print("concatenated_features shape :{}".format(concatenated_features.shape))

#     return concatenated_features, topk_indices, topk_values

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

    if args.temperature < 0.01:
        args.temperature = -1 # greedy
    max_context_length = getattr(
        model.config, 'max_position_embeddings', 2048)
    max_new_tokens = min(max_context_length - input_ids.shape[1], 512)
    
    with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                images=[[video_tensor], ['video']],
                # image_sizes=image_sizes,
                # question_ids=ques_ids,
                do_sample=True if args.temperature > 0 else False,
                temperature=args.temperature,
                top_p=args.top_p,
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
    # print(outputs)
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
    parser.add_argument("--top_p", type=float, default=0.9)

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
    print("Initialize baseline 3 using LLaVA_Hound_DPO !")
    tokenizer, model, processor, context_len = load_pretrained_model(model_path, args.model_base, model_name)


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
                # video_frames, sizes = load_video(video_path, num_frm=args.num_frames) # 实际上只抽取了num_frames对应的帧数      
                # spare_frames, frame_idx, sizes = read_images_in_video(video_path, args.num_frames)      
                # video_frames = get_most_change_frame(spare_frames, args.num_frames)
                # Run inference on the video and add the output to the list
                output = llava_hound_inference(question, conv_mode, model,
                                                tokenizer, processor['video'], video_path)
                # print(output)
                sample_set['pred'] = output
                output_list.append(sample_set)
                ans_file.write(json.dumps(sample_set) + "\n")
                break
        
        # if len(output_list) > sample_num:
        #     print("sample over !!")
        #     break
    ans_file.close()


# def
if __name__ == "__main__":
    args = parse_args()
    run_inference(args)