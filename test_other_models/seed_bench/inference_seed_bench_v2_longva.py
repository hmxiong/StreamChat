import sys
sys.path.append("/13390024681/llama/EfficientVideo/Ours")
import os
import json
import argparse

import torch
import numpy as np
import random
import pdb
import math
import copy
import numpy as np
from tqdm import tqdm
from PIL import Image
from longva.conversation import conv_templates, SeparatorStyle
from longva.constants import DEFAULT_IM_START_TOKEN, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_END_TOKEN, IMAGE_TOKEN_INDEX
from longva.mm_utils import get_model_name_from_path, tokenizer_image_token, KeywordsStoppingCriteria
from longva.model.builder import load_pretrained_model
# from decord import VideoReader, cpu
# from longva.model.language_model.llava_llama import LlavaLlamaForCausalLM
# from llava.eval.model_utils import load_video


# root directory of cc3m
cc3m_dir = "/13390024681/All_Data/SEED-Bench-H/cc3m-image"
# root directory of seed bench v2
seed_bench_v2_dir = "/13390024681/All_Data/SEED-Bench-H/SEED-Bench-2-image"

# seed = 0

# random.seed(seed)
# np.random.seed(seed)
# torch.manual_seed(seed)

# torch.cuda.manual_seed(seed)
# torch.cuda.manual_seed_all(seed)
# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = False

def filter_questions(data, level='L2', subpart='all', version='v2'):
    if level == "L1":
        valid_level_data = ['L1']
    elif level == "L2":
        valid_level_data = ['L1', 'L2']
    elif level == "L3":
        valid_level_data = ['L1', 'L2', 'L3']
    else:
        raise ValueError(f"Invalid level: {level}")
    data = [q for q in data if q["level"] in valid_level_data]

    if subpart in ['Single-Image & Text Comprehension', 'Multiple-Images & Text Comprehension', 'Video & Text Comprehension', 'Interleaved Image & Text Comprehension', 'Image Generation', 'Image & Text Generation']:
        valid_subgroup_data = subpart
    elif subpart == 'all':
        valid_subgroup_data = ['Single-Image & Text Comprehension', 'Multiple-Images & Text Comprehension', 'Video & Text Comprehension', 'Interleaved Image & Text Comprehension', 'Image Generation', 'Image & Text Generation']
    else:
        raise ValueError(f"Invalid subpart: {subpart}")
    data = [q for q in data if q["subpart"] in valid_subgroup_data]

    if version == 'v1':
        valid_version_data = ['v1']
    elif version == 'v2':
        valid_version_data = ['v1', 'v2']
    else:
        raise ValueError(f"Invalid version: {version}")
    data = [q for q in data if q["version"] in valid_version_data]

    return data

def get_model_response(model, tokenizer, processor, context_len, data_info):
    data_path, question, choices = data_info['data_path'], data_info['question'], data_info['choices']
    
    video = []
    
    if type(data_path) != list:
        print('data is not list type ')
        with Image.open(data_path) as raw_image:
            raw_image = raw_image.convert("RGB")
            video.append(raw_image)
    else:
        print('data is list ')
        # image = []
        for i in range(len(data_path)):
            with Image.open(data_path[i]) as raw_image:
                raw_image = raw_image.convert("RGB")
                video.append(raw_image)
    
    if "<img>" in question:
        question = question.replace("<img>", "")
    # if data_info['data_type'] == "Interleaved Image":
    #     if "<img>" in question:
    #         question = question.replace("<img>", "")
    # else:
    #     if "<img>" in question:
    #         question = question.replace("<img>", DEFAULT_IMAGE_TOKEN)
    
    all_losses = []
    with torch.no_grad():
            for cand in choices:
                qs = copy.deepcopy(question)
                
                if model.config.mm_use_im_start_end:
                    qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
                else:
                    qs = DEFAULT_IMAGE_TOKEN + '\n' + qs

                # conv_mode = "qwen_1_5"
                # args.conv_mode = conv_mode

                conv = conv_templates["qwen_1_5"].copy()
                conv.append_message(conv.roles[0], qs)
                conv.append_message(conv.roles[1], None)
                
                answer_input_id = torch.tensor(tokenizer(cand).input_ids).unsqueeze(0).cuda()
                prompt = conv.get_prompt()


                # video_tensor = video_processor.preprocess(video, return_tensors='pt')['pixel_values'][0].half().to(args.device)
                video_tensor = processor.preprocess(video, return_tensors="pt")["pixel_values"].to(model.device, dtype=torch.float16)
                # video_tensor = torch.cat([video_tensor[:len(video_tensor)//2], video_processor, video_tensor[len(video_tensor)//2:]], dim=0)
                # print(video_tensor.shape)
                input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(model.device)
                num_mask = answer_input_id.shape[1]
                labels = input_ids.clone()
                labels[:,:-1 * (num_mask)] = -100
                
                loss = model(input_ids, 
                             images=[video_tensor],
                             modalities=["video"], 
                             labels=labels).loss
                # print("loss:", loss)
                all_losses.append(loss.item())
                
    return all_losses

def is_integer_string(s):
    try:
        int(s)
        return True
    except ValueError:
        return False

def build_model(model_path):
    # Initialize the model
    print("Initialize LongVA !")
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, processor, context_len = load_pretrained_model(model_path, None , "llava_qwen")
    model = model.cuda()

    return model, tokenizer, processor, context_len

def run_inference(args, qa_anno, output_dir):
    model, tokenizer, processor, context_len = build_model(args.model)
    total_qa_num = len(qa_anno)
    answer_list = []
    output_f = open(os.path.join(output_dir, "results_longva.json"), "a")
    step = 0
    for qa_item in tqdm(qa_anno):

        data_info = {
            'question': qa_item['question'],
            'choices': [qa_item['choice_a'], qa_item['choice_b'], qa_item['choice_c'], qa_item['choice_d']],
        }

        if qa_item["data_source"] == 'cc3m':
            image_dir = cc3m_dir
        elif qa_item["data_source"] == 'SEED-Bench v2':
            image_dir = seed_bench_v2_dir
        else:
            raise ValueError("The data type is not valid.")
        
        if type(qa_item['data_id']) is list:
            data_path = [os.path.join(image_dir, path) for path in qa_item['data_id']]
        else:
            data_path = os.path.join(image_dir, qa_item['data_id'])
        data_info['data_path'] = data_path

        # losses: loss values of 4 choices, shape=[4]
        with torch.no_grad():
        #     losses = model(data_info)
            losses = get_model_response(model, tokenizer, processor, context_len, data_info)
            
        class_ranks = np.argsort(losses)
        pred_id = ['A', 'B', 'C', 'D'][class_ranks[0]]
        gt = qa_item['answer']
        answer_record = {
            'question_id': qa_item['question_id'],
            'prediction': pred_id,
            'gt':gt,
            'q_type_id':qa_item['question_type_id']
        }
        answer_list.append(answer_record)
        # output prediction record for each question
        output_f.write(json.dumps(answer_record) + "\n")
        step += 1

    print("evaluation finished! Calculating accuracy...")
    type_counts = {}
    correct_counts = {}

    for item in answer_list:
        pred, gt, data_type = item['prediction'], item['gt'], item['q_type_id']

        type_counts[data_type] = type_counts.get(data_type, 0) + 1
        if pred == gt:
            correct_counts[data_type] = correct_counts.get(data_type, 0) + 1

    print("Accuracy for each data type:")
    total_count = 0
    total_correct = 0
    for data_type in type_counts.keys():
        accuracy = correct_counts[data_type] / type_counts[data_type] * 100
        print(f"Data type {data_type}: {accuracy:.2f}%")

        total_count += type_counts[data_type]
        total_correct += correct_counts[data_type]

    total_accuracy = total_correct / total_count * 100
    print(f"Total accuracy: {total_accuracy:.2f}%")

def sing_inference_test_(args, qa_anno, output_dir):
    model, tokenizer, processor, context_len = build_model(args.model)
    total_qa_num = len(qa_anno)
    answer_list = []
    # output_f = open(os.path.join(output_dir, "results_longva.json"), "a")
    step = 0
    # for qa_item in tqdm(qa_anno):
    qa_item = {
            "answer": "B",
            "choice_a": "The man and woman in the image are both looking away from the camera.",
            "choice_b": "The woman's hair is black.",
            "choice_c": "The woman's dog is on the couch next to her in the image.",
            "choice_d": "There are two people in the image.",
            "data_id": [
                "task23/ICL_images/in_context_attribute_2/1.jpg",
                "task23/ICL_images/in_context_attribute_2/2.jpg",
                "task23/ICL_images/in_context_attribute_2/3.jpg"
            ],
            "data_source": "SEED-Bench v2",
            "data_type": "Interleaved Image",
            "level": "L2",
            "question": "<img>: The predominant color of the uniforms worn by the players is blue. <img>: The most notable color present in the woman's outfit is orange. <img>:",
            "question_id": "23_0",
            "question_type_id": 23,
            "subpart": "Interleaved Image & Text Comprehension",
            "version": "v2"
        }

    data_info = {
        'question': qa_item['question'],
        'choices': [qa_item['choice_a'], qa_item['choice_b'], qa_item['choice_c'], qa_item['choice_d']],
        'data_type': qa_item['data_type']
    }

    if qa_item["data_source"] == 'cc3m':
        image_dir = cc3m_dir
    elif qa_item["data_source"] == 'SEED-Bench v2':
        image_dir = seed_bench_v2_dir
    else:
        raise ValueError("The data type is not valid.")
    
    if type(qa_item['data_id']) is list:
        data_path = [os.path.join(image_dir, path) for path in qa_item['data_id']]
    else:
        data_path = os.path.join(image_dir, qa_item['data_id'])
    data_info['data_path'] = data_path

    # losses: loss values of 4 choices, shape=[4]
    with torch.no_grad():
    #     losses = model(data_info)
        losses = get_model_response(model, tokenizer, processor, context_len, data_info)
    
    print(losses)
    class_ranks = np.argsort(losses)
        
if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Arg Parser')
    parser.add_argument('--model', type=str, default='/13390024681/All_Model_Zoo/LongVA-7B-DPO')
    parser.add_argument('--anno_path', type=str, default='/13390024681/All_Data/SEED-Bench-H/SEED-Bench_v2_level1_2_3.json')
    parser.add_argument('--output_dir', type=str, default='/13390024681/llama/EfficientVideo/Ours/output/SEED_Bench')
    parser.add_argument('--evaluate_level', type=str, default='L2')
    parser.add_argument('--evaluate_part', type=str, default='all')
    parser.add_argument('--evaluate_version', type=str, default='v2')
    args = parser.parse_args()
    
    args = parser.parse_args()

    qa_anno = json.load(open(args.anno_path, 'rb'))
    if 'questions' in qa_anno.keys():
        qa_anno = qa_anno['questions']
    qa_anno = filter_questions(qa_anno, args.evaluate_level, args.evaluate_part, args.evaluate_version)
    # pdb.set_trace()

    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)

    print(f'evaluating.. {args.model}')
    # The interface for testing MLLMs
    # model = build_model(args.model).cuda()
    run_inference(args, qa_anno, args.output_dir)
    # sing_inference_test_(args, qa_anno, args.output_dir)