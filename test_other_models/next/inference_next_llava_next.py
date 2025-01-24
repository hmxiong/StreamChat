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
from llavanext.conversation import conv_templates, SeparatorStyle
from llavanext.constants import DEFAULT_IM_START_TOKEN, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_END_TOKEN, IMAGE_TOKEN_INDEX
from llavanext.mm_utils import get_model_name_from_path, tokenizer_image_token, KeywordsStoppingCriteria
from llavanext.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path
from llavanext.model.builder import load_pretrained_model
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

def get_model_output(model, video_processor, tokenizer, video_frames, sizes, qs, args):
    if model.config.mm_use_im_start_end:
        qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
    else:
        qs = DEFAULT_IMAGE_TOKEN + '\n' + qs

    # conv_mode = "qwen_1_5"
    # args.conv_mode = conv_mode

    conv = conv_templates["llava_llama_3"].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()


    # video_tensor = video_processor.preprocess(video, return_tensors='pt')['pixel_values'][0].half().to(args.device)
    # video_tensor = video_processor.preprocess(video, return_tensors="pt")["pixel_values"].to(model.device, dtype=torch.float16)
    # video_tensor = torch.cat([video_tensor[:len(video_tensor)//2], video_processor, video_tensor[len(video_tensor)//2:]], dim=0)
    # print(video_tensor.shape)
    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(args.device)
    image_tensor = process_images(video_frames, video_processor, model.config)  
    # input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt')

    # stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
    # keywords = [stop_str]
    # stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)
    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            images=image_tensor.to(dtype=torch.float16, device='cuda', non_blocking=True),
            image_sizes=sizes,
            # question_ids=ques_ids,
            # modalities="image",
            do_sample=True,
            temperature=0.1,
            top_p=None,
            num_beams=1,
            max_new_tokens=256,
            use_cache=False)
        
    # with torch.inference_mode():
    #     output_ids = model.generate(
    #         input_ids,
    #         images=[image_tensor],
    #         modalities=["video"],
    #         do_sample=True,
    #         temperature=0.1,
    #         max_new_tokens=256,
    #         num_beams=1,
    #         use_cache=False)

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
    # print(outputs)
    return outputs


def run_inference(args):
    """
    Run inference on ActivityNet QA DataSet using the Video-ChatGPT model.

    Args:
        args: Command-line arguments.
    """
    # Initialize the model
    print("Initialize LLaVA-NExT !")
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
    elif "ActiveNet" in args.video_dir:
        mode = "ActiveNet"
    elif "NExT" in args.video_dir:
        mode = "NEXT"
    else:
        mode = "Others"
        
    video_formats = ['.mp4', '.avi', '.mov', '.mkv']

    video_path_dict = {
        "/13390024681/All_Data/Streaming_final":["Cooking_show" ,"Comedy_drama" ,"Apple_TV"], 
        "/13390024681/All_Data/Supplement_1":["Cooking", "Metalworking"]
        }

    def find_key_by_category(category, data_dict):
        for key, categories in data_dict.items():
            if category in categories:
                return key
        return None
    
    # Iterate over each sample in the ground truth file
    index = 0
    for sample in tqdm(gt_questions, desc="LLaVA-NExT Inference for:{}".format(mode)):
        
        video_name = sample['video']
        question = sample['question']
        answer = sample['answer']
        id = sample['qid']
        
        index += 1
        
        sample_set = {'id': id, 'question': question, 'answer': answer}

        # Load the video file
        temp_path = os.path.join(args.video_dir, f"{video_name}.mp4")
        
        if os.path.exists(temp_path):
            video_path = temp_path
            # try:
            # Run inference on the video and add the output to the list
            video_frame, video_size = load_video(video_path, num_frm=8)
            output = get_model_output(model, processor, tokenizer, video_frame, video_size, question, args)
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
