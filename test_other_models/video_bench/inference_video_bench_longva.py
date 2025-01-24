import sys
sys.path.append("/13390024681/llama/EfficientVideo/Ours")
# sys.path.append("/13390024681/llama/EfficientVideo/Ours")
import math
import os
import argparse
import json
import torch
import transformers
import numpy as np
from tqdm import tqdm
from longva.conversation import conv_templates, SeparatorStyle
from longva.constants import DEFAULT_IM_START_TOKEN, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_END_TOKEN, IMAGE_TOKEN_INDEX
from longva.mm_utils import get_model_name_from_path, tokenizer_image_token, KeywordsStoppingCriteria
from longva.model.builder import load_pretrained_model
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
    parser.add_argument('--cache_dir', help='', required=False)
    # parser.add_argument('--video_dir', help='Directory containing video files.', required=True)
    # parser.add_argument('--gt_file_question', help='Path to the ground truth file containing question.', required=True)
    # parser.add_argument('--gt_file_answers', help='Path to the ground truth file containing answers.', required=False)
    # parser.add_argument('--output_dir', help='Directory to save the model results JSON.', required=True)
    # parser.add_argument('--output_name', help='Name of the file for storing results JSON.', required=True)
    parser.add_argument("--num_chunks", type=int, default=1)
    parser.add_argument("--num_frame", type=int, default=8)
    parser.add_argument("--chunk_idx", type=int, default=0)
    parser.add_argument("--device", type=str, required=False, default='cuda:0')
    parser.add_argument('--model_base', help='', default=None, type=str, required=False)
    parser.add_argument("--model_max_length", type=int, required=False, default=2048)

    return parser.parse_args()

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
            max_new_tokens=256,
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
    # print(outputs)
    return outputs


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

    Eval_QA_root = "/13390024681/llama/EfficientVideo/Ours/test_other_models/video_bench/Video-Bench"
    Eval_Video_root = "/13390024681/All_Data/Video-Bench"
    dataset_qajson = {
        "Ucfcrime": f"{Eval_QA_root}/Eval_QA/Ucfcrime_QA_new.json",
        "Youcook2": f"{Eval_QA_root}/Eval_QA/Youcook2_QA_new.json",
        "TVQA": f"{Eval_QA_root}/Eval_QA/TVQA_QA_new.json",
        "MSVD": f"{Eval_QA_root}/Eval_QA/MSVD_QA_new.json",
        "MSRVTT": f"{Eval_QA_root}/Eval_QA/MSRVTT_QA_new.json",
        "Driving-decision-making": f"{Eval_QA_root}/Eval_QA/Driving-decision-making_QA_new.json",
        "NBA": f"{Eval_QA_root}/Eval_QA/NBA_QA_new.json",
        "SQA3D": f"{Eval_QA_root}/Eval_QA/SQA3D_QA_new.json",
        "Driving-exam": f"{Eval_QA_root}/Eval_QA/Driving-exam_QA_new.json",
        "MV": f"{Eval_QA_root}/Eval_QA/MV_QA_new.json",
        "MOT": f"{Eval_QA_root}/Eval_QA/MOT_QA_new.json",
        "ActivityNet": f"{Eval_QA_root}/Eval_QA/ActivityNet_QA_new.json",
        "TGIF": f"{Eval_QA_root}/Eval_QA/TGIF_QA_new.json"
        }
    dataset_name_list = list(dataset_qajson.keys())
    print(dataset_name_list)
    
    index = 0
    num_frame = args.num_frame
    
    for dataset_name in tqdm(dataset_name_list, desc="LongVA Inference for Video_Bench in {}".format(num_frame)):
        qa_json = dataset_qajson[dataset_name]
        print(f'Dataset name:{dataset_name}, {qa_json=}!')
        with open(qa_json, 'r', encoding='utf-8') as f:
            all_annotations = json.load(f)
        
        # all_annotations = get_chunk(all_annotations, args.num_chunks, args.chunk_idx)
        
        eval_dict = {}
        for q_id, item in tqdm(all_annotations.items()):
            video_id = item['video_id']
            question = item['question'] 
            if len(item['choices']) == 6:
                question += f"Choices: A.{item['choices']['A']} B.{item['choices']['B']} C.{item['choices']['C']} D.{item['choices']['D']} E.{item['choices']['E']} F.{item['choices']['F']} \n Among the six options A, B, C, D, E, F above, the one closest to the correct answer is:"
                candidates = ['A', 'B', 'C', 'D', 'E', 'F']
                candidates_long = [f" A.{item['choices']['A']}", f"B.{item['choices']['B']}", f"C.{item['choices']['C']}", f"D.{item['choices']['D']}", f"E.{item['choices']['E']}", f"F.{item['choices']['F']}"]
            elif len(item['choices']) == 5:
                question += f" A.{item['choices']['A']} B.{item['choices']['B']} C.{item['choices']['C']} D.{item['choices']['D']} E.{item['choices']['E']} \n Among the five options A, B, C, D, E above, the one closest to the correct answer is: "
                candidates = ['A', 'B', 'C', 'D', 'E']
                candidates_long = [f" A.{item['choices']['A']}", f"B.{item['choices']['B']}", f"C.{item['choices']['C']}", f"D.{item['choices']['D']}", f"E.{item['choices']['E']}"]
            elif len(item['choices']) == 4:
                question += f" A.{item['choices']['A']} B.{item['choices']['B']} C.{item['choices']['C']} D.{item['choices']['D']} \n Among the four options A, B, C, D above, the one closest to the correct answer is:"
                candidates = ['A', 'B', 'C', 'D']
                candidates_long = [f" A.{item['choices']['A']}", f"B.{item['choices']['B']}", f"C.{item['choices']['C']}", f"D.{item['choices']['D']}"]
            elif len(item['choices']) == 3:
                question += f" A.{item['choices']['A']} B.{item['choices']['B']} C.{item['choices']['C']} \n Among the three options A, B, C above, the one closest to the correct answer is: "
                candidates = ['A', 'B', 'C']
                candidates_long = [f" A.{item['choices']['A']}", f"B.{item['choices']['B']}", f"C.{item['choices']['C']}"]
            elif len(item['choices']) == 2:
                question += f" A.{item['choices']['A']} B.{item['choices']['B']} \n Among the two options A, B above, the one closest to the correct answer is: "
                candidates = ['A', 'B']
                candidates_long = [f" A.{item['choices']['A']}", f"B.{item['choices']['B']}"]
            vid_rela_path = item['vid_path']
            video_path = os.path.join(Eval_Video_root, vid_rela_path)
            
            if os.path.exists(video_path):
                # video_path = temp_path
                # try:
                # Run inference on the video and add the output to the list
                video_frame, video_size = load_video(video_path, num_frm=num_frame)
                output = get_model_output(model, processor, tokenizer, video_frame, question, args)
                eval_dict[q_id] = {
                    'video_id': video_id,
                    'question': question,
                    'output_sequence': output
                }  
                print(f'q_id:{q_id}, output:{output}!\n')

        # eval results
        if not os.path.exists(f"/13390024681/llama/EfficientVideo/Ours/output/Video_Bench/LongVA_{num_frame}"):
            os.mkdir(f"/13390024681/llama/EfficientVideo/Ours/output/Video_Bench/LongVA_{num_frame}")
        
        # # eval results
        # if not os.path.exists(f"/13390024681/llama/EfficientVideo/Ours/output/Video_Bench/LongVA_{num_frame}/{dataset_name}"):
        #     os.mkdir(f"/13390024681/llama/EfficientVideo/Ours/output/Video_Bench/LongVA_{num_frame}/{dataset_name}")
            
        # eval_dataset_json = f'/13390024681/llama/EfficientVideo/Ours/output/Video_Bench/LongVA_{num_frame}/{dataset_name}/{args.num_chunks}_{args.chunk_idx}.json'
        # with open(eval_dataset_json, 'w', encoding='utf-8') as f:
        #     json.dump(eval_dict, f, indent=2)
        
        eval_dataset_json = f'/13390024681/llama/EfficientVideo/Ours/output/Video_Bench/LongVA_{num_frame}/{dataset_name}_eval.json'
        with open(eval_dataset_json, 'w', encoding='utf-8') as f:
            json.dump(eval_dict, f, indent=2)
    # Save the output list to a JSON file
    # with open(os.path.join(args.output_dir, f"{args.output_name}.json"), 'w') as file:
    #     json.dump(output_list, file)


if __name__ == "__main__":
    args = parse_args()
    run_inference(args)
