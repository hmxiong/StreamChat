import argparse
import torch

from llamavid.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llamavid.conversation import conv_templates, SeparatorStyle
from llamavid.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, \
                         LlamaConfig, LlamaModel, LlamaForCausalLM
import json
import os
import re
import ast
import math
from tqdm import tqdm
import pandas as pd
from decord import VideoReader, cpu


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
    parser.add_argument('--llama3', help='Path to the llama3 model file.', required=True)
    parser.add_argument('--gt_file', help='Path to the ground truth file containing question.', required=True)
    parser.add_argument('--output_dir', help='Directory to save the model results JSON.', required=True)
    parser.add_argument('--output_name', help='Name of the file for storing results JSON.', required=True)
    parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--data_set", type=str, default="msvd")
    parser.add_argument("--conv-mode", type=str, default=None)
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--model-max-length", type=int, default=None)
    parser.add_argument("--debug", type=bool, default=False)

    return parser.parse_args()


def load_video(video_path):
    vr = VideoReader(video_path, ctx=cpu(0))
    total_frame_num = len(vr)
    fps = round(vr.get_avg_fps())
    frame_idx = [i for i in range(0, len(vr), fps)]
    spare_frames = vr.get_batch(frame_idx).asnumpy()
    return spare_frames

def prepare_caption(caption_list):
    captions = []
    for index, single_caption in enumerate(caption_list):
        single_caption  = str(index+1) + ":" + single_caption
        captions.append(single_caption)
    
    caption = " ".join(captions)
    return caption

def prepare_prompt(tokenizer:AutoTokenizer, messages:list):
    # tokens = []
    complete_message = []
    complete_message.append("<|begin_of_text|>")
    for messgae in messages:
        # tokens.append(tokenizer("<|start_header_id|>"))
        # tokens.append()
        complete_message.append("<|start_header_id|>")
        complete_message.append(messgae["role"])
        complete_message.append("<|end_header_id|>")
        complete_message.append("\n\n")
        complete_message.append(messgae["content"])
        complete_message.append("<|eot_id|>")
    
    # complete_message.append("<|eot_id|>")
    # complete_message.append("<|start_header_id|>")
    # complete_message.append("assistant")
    # complete_message.append("<|end_header_id|>")
    # complete_message.append("\n\n")
    
    
    complete_message = " ".join(complete_message)
    message_ids = tokenizer(complete_message)
    # print(message_ids)
    # message_token
    # print(complete_message)
    # assert 1==2
    # message_ids = torch.tensor(message_ids, dtype=torch.long)
    return complete_message, message_ids

def run_inference_video_chat(args):
    """
    Run inference on VideoChat QA DataSet using the Video-ChatGPT model.

    Args:
        args: Command-line arguments.
    """
    # Initialize the model
    debug = args.debug
    print("Loading LLaMA-VID model !!!")
    model_name = get_model_name_from_path(args.model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(args.model_path, args.model_base, model_name, args.model_max_length)
    print("LLaMA-VID load finish !!")
    print("loading LLaMa3-8B for captioning !!")
    llama_path = args.llama3
    kwargs = {"device_map": "auto"}
    kwargs['torch_dtype'] = torch.float16
    llama_config = LlamaConfig.from_pretrained(llama_path)
    llama_tokenizer = AutoTokenizer.from_pretrained(llama_path, use_fast=True)
    llama_model = LlamaForCausalLM.from_pretrained(llama_path, low_cpu_mem_usage=True, config=llama_config, **kwargs)
    llama_tokenizer.pad_token = llama_tokenizer.eos_token
    print(llama_model.device)
    print("LLaMa3-8B laod finish !!")

    # llama_vid_prompt = [
    #     "What is the background of this movie?",
    #     "What is this movie talking about?",
    #     # "What is the main character of this movie?"
    # ]
    # Load both ground truth file containing questions and answers
    with open(args.gt_file) as file:
        gt_questions = json.load(file)
    gt_questions = get_chunk(gt_questions, args.num_chunks, args.chunk_idx)

    # Create the output directory if it doesn't exist
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    video_formats = ['.mp4', '.avi', '.mov', '.mkv']
    if args.num_chunks > 1:
        output_name = f"{args.num_chunks}_{args.chunk_idx}"
    else:
        output_name = args.output_name
    answers_file = os.path.join(args.output_dir, f"{output_name}.json")
    ans_file = open(answers_file, "w")

    for sample in tqdm(gt_questions, desc="LLaMA-VID with LLaMA-3 in {}:".format(args.data_set)):
        # video_name = sample['video_id']
        # question = sample['question']
        # id = sample['id']
        # answer = sample['answer']
        video_name = sample['video']
        question = sample['question']
        answer = sample['answer']
        id = sample['question_id']

        sample_set = {'id': id, 'question': question, 'answer': answer}

        # Load the video file
        for fmt in video_formats:  # Added this line
            temp_path = os.path.join(args.video_dir, f"{video_name}")
            if os.path.exists(temp_path):
                video_path = temp_path
                break

        # Check if the video exists
        if os.path.exists(video_path):
            video = load_video(video_path)
            video = image_processor.preprocess(video, return_tensors='pt')['pixel_values'].half().cuda()
            video = [video]

        # try:
            # Run inference on the video and add the output to the list
        captions = []
        # llama_vid_prompt.append(question)
        llama_vid_prompt = [
        # "What is the background of this movie?",
        "What can you see from this video? What's the relationship between them?",
        "Describe this video as much detail as possible."
        # "What is this video talking about?",
        # "What is the main character of this movie?"
        ]
        llama_vid_prompt.append(question)
        assert len(llama_vid_prompt) == 3
        for pe_prompt in llama_vid_prompt:
            qs = pe_prompt
            if model.config.mm_use_im_start_end:
                qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
            else:
                qs = DEFAULT_IMAGE_TOKEN + '\n' + qs

            conv = conv_templates[args.conv_mode].copy()
            conv.append_message(conv.roles[0], qs)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()

            input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()

            stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
            keywords = [stop_str]
            stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)

            cur_prompt = pe_prompt
            with torch.inference_mode():
                model.update_prompt([[cur_prompt]])
                output_ids = model.generate(
                    input_ids,
                    images=video,
                    do_sample=True,
                    temperature=0.1,
                    max_new_tokens=1024,
                    use_cache=True,
                    stopping_criteria=[stopping_criteria])

            input_token_len = input_ids.shape[1]
            n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
            if n_diff_input_output > 0:
                print(f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')
            outputs = tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0]
            outputs = outputs.strip()
            if outputs.endswith(stop_str):
                outputs = outputs[:-len(stop_str)]
            outputs = outputs.strip()
            captions.append(outputs)
            # print("llama-vid:{}".format(outputs))
            
        caption = prepare_caption(captions)
        # print("llama-vid caption:{}".format(caption))
        llama3_prompt =[
                    {
                        "role": "system",
                        "content":
                            "You are an intelligent chatbot designed for generating safe, complete answers based on instructions provided by user."
                            # "These information come from another multi-modal chatbot, which needs to watch a video and get corresponding answers based on prompts with different attributes."
                    },
                    {
                        "role": "user",
                        "content":
                            "The video-based question and captin pair:\n"
                            f"Question: {question}\n"
                            f"Captions: {caption}\n"
                            "Please generate the answer of the question based on the information provided."
                            "Please generate the response in the form of a Python dictionary string with no keys, where the generated string is between '{' and '}'."
                            "DO NOT PROVIDE ANY OTHER OUTPUT TEXT OR EXPLANATION. Please make supplementary summaries in order to provide a richer and more comprehensive answer. Only provide the right answer string. "
                            "For example, your response should look like this: { YOUR ANSWER }."
                            # "Please generate the response in the form of a Python dictionary string with keys 'llama_pred', where value of 'llama_pred' is the strings generated."
                            # "DO NOT PROVIDE ANY OTHER OUTPUT TEXT OR EXPLANATION. Only provide the right answer string. "
                            # "For example, your response should look like this: {'llama_pred': 'YOUR ANSWER' }."
                    }
                ]
        
        complete_message, message_ids = prepare_prompt(llama_tokenizer, llama3_prompt)
        # print("llama3_messgae:{}".format(complete_message))
        
        ids = torch.tensor(message_ids['input_ids'], dtype=torch.long).to(llama_model.device)
        attention_mask = torch.tensor(message_ids['attention_mask']).to(llama_model.device).unsqueeze(0)
        
        embeddings = llama_model.model.embed_tokens(ids).unsqueeze(0).to(dtype=torch.float16)
        
        with torch.inference_mode():
            output_ids = llama_model.generate(
                inputs_embeds = embeddings,
                attention_mask = attention_mask,
                pad_token_id=llama_tokenizer.eos_token_id,
                do_sample=True,
                temperature=0.1,
                # top_p=args.top_p,
                max_new_tokens=512,
                # use_cache=True
            )
        # print(answer)
        out_text = llama_tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]
        # print("original out:{}".format(out_text))
        match = re.search(r'\{(.*?)\}', out_text)

        # 检查是否有匹配结果
        if match:
            extracted_text = match.group(1)
        # matches = re.findall(r'\{.*?\}', out_text)
        # for match in matches:
        #     result_dict = ast.literal_eval(match)
        
        # print("llama3 answer:{}".format(result_dict['llama_pred']))
        # if debug:
        # print("llama3 prediction:{}".format(extracted_text))
            
        sample_set['pred'] = extracted_text
        sample_set['caption'] = caption
        ans_file.write(json.dumps(sample_set) + "\n")
        ans_file.flush()

    ans_file.close()

def run_inference_intent(args):
    """
    Run inference on Intent QA DataSet using our model.

    Args:
        args: Command-line arguments.
    """
    # Initialize the model
    debug = args.debug
    print("Loading LLaMA-VID model !!!")
    model_name = get_model_name_from_path(args.model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(args.model_path, args.model_base, model_name, args.model_max_length)
    print("LLaMA-VID load finish !!")
    print("loading LLaMa3-8B for captioning !!")
    llama_path = args.llama3
    kwargs = {"device_map": "auto"}
    kwargs['torch_dtype'] = torch.float16
    llama_config = LlamaConfig.from_pretrained(llama_path)
    llama_tokenizer = AutoTokenizer.from_pretrained(llama_path, use_fast=True)
    llama_model = LlamaForCausalLM.from_pretrained(llama_path, low_cpu_mem_usage=True, config=llama_config, **kwargs)
    llama_tokenizer.pad_token = llama_tokenizer.eos_token
    print(llama_model.device)
    print("LLaMa3-8B laod finish !!")

    # llama_vid_prompt = [
    #     "What is the background of this movie?",
    #     "What is this movie talking about?",
    #     # "What is the main character of this movie?"
    # ]
    # Load both ground truth file containing questions and answers
    # with open("/13390024681/All_Data/nextqa/map_vid_vidorID.json") as file:
    #     map_file = json.load(file)
        
    gt_questions = pd.read_csv(args.gt_file)
    gt_questions = get_chunk(gt_questions, args.num_chunks, args.chunk_idx)

    # Create the output directory if it doesn't exist
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    video_formats = ['.mp4', '.avi', '.mov', '.mkv']
    if args.num_chunks > 1:
        output_name = f"{args.num_chunks}_{args.chunk_idx}"
    else:
        output_name = args.output_name
    answers_file = os.path.join(args.output_dir, f"{output_name}.json")
    ans_file = open(answers_file, "w")

    for sample in tqdm(gt_questions.iterrows(), desc="LLaMA-VID with LLaMA-3 in Intent-QA"):
        # video_name = sample['video_id']
        # question = sample['question']
        # id = sample['id']
        # answer = sample['answer']
        # print(sample)
        if isinstance(sample, tuple):
            sample = sample[-1]
        # print(sample)
        video_name = str(sample['video'])
        question, truth = sample['question'], sample['answer']
        qid, q_type = sample['qid'], sample['type']
        choices = [sample['a0'], sample['a1'], sample['a2'], sample['a3'], sample['a4']]
        quid = f'{video_name}_{qid}'
        
        # video_name = map_file[uid]

        sample_set = {'id': quid, 'question': question, "truth":truth, "q_type":q_type}

        # Load the video file
        for fmt in video_formats:  # Added this line
            temp_path = os.path.join(args.video_dir, f"{video_name}.mp4")
            if os.path.exists(temp_path):
                video_path = temp_path
                break

        # Check if the video exists
        if os.path.exists(video_path):
            video = load_video(video_path)
            video = image_processor.preprocess(video, return_tensors='pt')['pixel_values'].half().cuda()
            video = [video]

        # try:
            # Run inference on the video and add the output to the list
        captions = []
        # llama_vid_prompt.append(question)
        llama_vid_prompt = [
        # "What is the background of this movie?",
        "What can you see from this video? What's the relationship between them?",
        "This video you see is mainly about a person or object doing something .Describe this video as much detail as possible."
        # "What is this video talking about?",
        # "What is the main character of this movie?"
        ]
        llama_vid_prompt.append(question)
        assert len(llama_vid_prompt) == 3
        for pe_prompt in llama_vid_prompt:
            qs = pe_prompt
            if model.config.mm_use_im_start_end:
                qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
            else:
                qs = DEFAULT_IMAGE_TOKEN + '\n' + qs

            conv = conv_templates[args.conv_mode].copy()
            conv.append_message(conv.roles[0], qs)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()

            input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()

            stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
            keywords = [stop_str]
            stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)

            cur_prompt = pe_prompt
            with torch.inference_mode():
                model.update_prompt([[cur_prompt]])
                output_ids = model.generate(
                    input_ids,
                    images=video,
                    do_sample=True,
                    temperature=0.1,
                    max_new_tokens=1024,
                    use_cache=True,
                    stopping_criteria=[stopping_criteria])

            input_token_len = input_ids.shape[1]
            n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
            if n_diff_input_output > 0:
                print(f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')
            outputs = tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0]
            outputs = outputs.strip()
            if outputs.endswith(stop_str):
                outputs = outputs[:-len(stop_str)]
            outputs = outputs.strip()
            captions.append(outputs)
            # print("llama-vid:{}".format(outputs))
            
        caption = prepare_caption(captions)
        # print("llama-vid caption:{}".format(caption))
        llama3_prompt =[
                    {
                        "role": "system",
                        "content":
                            "You are an intelligent chatbot designed for generating safe, complete answers based on instructions provided by user."
                            # "These information come from another multi-modal chatbot, which needs to watch a video and get corresponding answers based on prompts with different attributes."
                    },
                    {
                        "role": "user",
                        "content":
                            "The video-based question and captin pair:\n"
                            f"Question: {question}\n"
                            f"Captions: {caption}\n"
                            "Here are the five answer choices for the question:"
                            f"0:{choices[0]};1:{choices[1]};2.{choices[2]};3.{choices[3]};4.{choices[4]}."
                            "Please choose the answer you think is correct among the above options in the form of a Python dictionary string with no keys, where the generated string is between '{' and '}'."
                            "DO NOT PROVIDE ANY OTHER OUTPUT TEXT OR EXPLANATION. You just need to generate the number corresponding to the correct option. "
                            "For example, if you think the right answer is option 3 and your response should look like this: { 3 }."
                    
                    }
                ]
        
        complete_message, message_ids = prepare_prompt(llama_tokenizer, llama3_prompt)
        # print("llama3_messgae:{}".format(complete_message))
        
        ids = torch.tensor(message_ids['input_ids'], dtype=torch.long).to(llama_model.device)
        attention_mask = torch.tensor(message_ids['attention_mask']).to(llama_model.device).unsqueeze(0)
        
        embeddings = llama_model.model.embed_tokens(ids).unsqueeze(0).to(dtype=torch.float16)
        
        with torch.inference_mode():
            output_ids = llama_model.generate(
                inputs_embeds = embeddings,
                attention_mask = attention_mask,
                pad_token_id=llama_tokenizer.eos_token_id,
                do_sample=True,
                temperature=0.1,
                # top_p=args.top_p,
                max_new_tokens=100,
                # use_cache=True
            )
        # print(answer)
        out_text = llama_tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]
        # print("original out:{}".format(out_text))
        # match = re.search(r'\{(.*?)\}', out_text)
        match = re.search(r'{\s*(\d+)\s*}', out_text)
        
        # 检查是否有匹配结果
        if match:
            extracted_text = int(match.group(1))
            # print(extracted_text)
        # matches = re.findall(r'\{.*?\}', out_text)
        # for match in matches:
        #     result_dict = ast.literal_eval(match)
        
        # print("llama3 answer:{}".format(result_dict['llama_pred']))
        # if debug:
        # print("llama3 prediction:{}".format(extracted_text))
            
        sample_set['correct_answer'] = extracted_text
        sample_set['caption'] = caption
        ans_file.write(json.dumps(sample_set) + "\n")
        ans_file.flush()

    ans_file.close()
    pass

def run_inference_next(args):
    """
    Run inference on NExT QA DataSet using our model.

    Args:
        args: Command-line arguments.
    """
    # Initialize the model
    debug = args.debug
    print("Loading LLaMA-VID model !!!")
    model_name = get_model_name_from_path(args.model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(args.model_path, args.model_base, model_name, args.model_max_length)
    print("LLaMA-VID load finish !!")
    print("loading LLaMa3-8B for captioning !!")
    llama_path = args.llama3
    kwargs = {"device_map": "auto"}
    kwargs['torch_dtype'] = torch.float16
    llama_config = LlamaConfig.from_pretrained(llama_path)
    llama_tokenizer = AutoTokenizer.from_pretrained(llama_path, use_fast=True)
    llama_model = LlamaForCausalLM.from_pretrained(llama_path, low_cpu_mem_usage=True, config=llama_config, **kwargs)
    llama_tokenizer.pad_token = llama_tokenizer.eos_token
    print(llama_model.device)
    print("LLaMa3-8B laod finish !!")

    # llama_vid_prompt = [
    #     "What is the background of this movie?",
    #     "What is this movie talking about?",
    #     # "What is the main character of this movie?"
    # ]
    # Load both ground truth file containing questions and answers
    with open("/13390024681/All_Data/nextqa/map_vid_vidorID.json") as file:
        map_file = json.load(file)
        
    gt_questions = pd.read_csv(args.gt_file)
    gt_questions = get_chunk(gt_questions, args.num_chunks, args.chunk_idx)

    # Create the output directory if it doesn't exist
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    video_formats = ['.mp4', '.avi', '.mov', '.mkv']
    if args.num_chunks > 1:
        output_name = f"{args.num_chunks}_{args.chunk_idx}"
    else:
        output_name = args.output_name
    answers_file = os.path.join(args.output_dir, f"{output_name}.json")
    ans_file = open(answers_file, "w")

    for sample in tqdm(gt_questions.iterrows(), desc="LLaMA-VID with LLaMA-3 in NExT-QA"):
        # video_name = sample['video_id']
        # question = sample['question']
        # id = sample['id']
        # answer = sample['answer']
        # print(sample)
        if isinstance(sample, tuple):
            sample = sample[-1]
        # print(sample)
        uid = str(sample['video'])
        question, truth = sample['question'], sample['answer']
        qid, q_type = sample['qid'], sample['type']
        choices = [sample['a0'], sample['a1'], sample['a2'], sample['a3'], sample['a4']]
        quid = f'{uid}_{qid}'
        
        video_name = map_file[uid]

        sample_set = {'id': quid, 'question': question, "truth":truth, "q_type":q_type}

        # Load the video file
        for fmt in video_formats:  # Added this line
            temp_path = os.path.join(args.video_dir, f"{video_name}.mp4")
            if os.path.exists(temp_path):
                video_path = temp_path
                break

        # Check if the video exists
        if os.path.exists(video_path):
            video = load_video(video_path)
            video = image_processor.preprocess(video, return_tensors='pt')['pixel_values'].half().cuda()
            video = [video]

        # try:
            # Run inference on the video and add the output to the list
        captions = []
        # llama_vid_prompt.append(question)
        llama_vid_prompt = [
        # "What is the background of this movie?",
        "What can you see from this video? What's the relationship between them?",
        "Describe this video as much detail as possible."
        # "What is this video talking about?",
        # "What is the main character of this movie?"
        ]
        llama_vid_prompt.append(question)
        assert len(llama_vid_prompt) == 3
        for pe_prompt in llama_vid_prompt:
            qs = pe_prompt
            if model.config.mm_use_im_start_end:
                qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
            else:
                qs = DEFAULT_IMAGE_TOKEN + '\n' + qs

            conv = conv_templates[args.conv_mode].copy()
            conv.append_message(conv.roles[0], qs)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()

            input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()

            stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
            keywords = [stop_str]
            stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)

            cur_prompt = pe_prompt
            with torch.inference_mode():
                model.update_prompt([[cur_prompt]])
                output_ids = model.generate(
                    input_ids,
                    images=video,
                    do_sample=True,
                    temperature=0.1,
                    max_new_tokens=1024,
                    use_cache=True,
                    stopping_criteria=[stopping_criteria])

            input_token_len = input_ids.shape[1]
            n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
            if n_diff_input_output > 0:
                print(f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')
            outputs = tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0]
            outputs = outputs.strip()
            if outputs.endswith(stop_str):
                outputs = outputs[:-len(stop_str)]
            outputs = outputs.strip()
            captions.append(outputs)
            # print("llama-vid:{}".format(outputs))
            
        caption = prepare_caption(captions)
        # print("llama-vid caption:{}".format(caption))
        llama3_prompt =[
                    {
                        "role": "system",
                        "content":
                            "You are an intelligent chatbot designed for generating safe, complete answers based on instructions provided by user."
                            # "These information come from another multi-modal chatbot, which needs to watch a video and get corresponding answers based on prompts with different attributes."
                    },
                    {
                        "role": "user",
                        "content":
                            "The video-based question and captin pair:\n"
                            f"Question: {question}\n"
                            f"Captions: {caption}\n"
                            "Here are the five answer choices for the question:"
                            f"0:{choices[0]};1:{choices[1]};2.{choices[2]};3.{choices[3]};4.{choices[4]}."
                            "Please choose the answer you think is correct among the above options in the form of a Python dictionary string with no keys, where the generated string is between '{' and '}'."
                            "DO NOT PROVIDE ANY OTHER OUTPUT TEXT OR EXPLANATION. You just need to generate the number corresponding to the correct option. "
                            "For example, if you think the right answer is option 3 and your response should look like this: { 3 }."
                    
                    }
                ]
        
        complete_message, message_ids = prepare_prompt(llama_tokenizer, llama3_prompt)
        # print("llama3_messgae:{}".format(complete_message))
        
        ids = torch.tensor(message_ids['input_ids'], dtype=torch.long).to(llama_model.device)
        attention_mask = torch.tensor(message_ids['attention_mask']).to(llama_model.device).unsqueeze(0)
        
        embeddings = llama_model.model.embed_tokens(ids).unsqueeze(0).to(dtype=torch.float16)
        
        with torch.inference_mode():
            output_ids = llama_model.generate(
                inputs_embeds = embeddings,
                attention_mask = attention_mask,
                pad_token_id=llama_tokenizer.eos_token_id,
                do_sample=True,
                temperature=0.1,
                # top_p=args.top_p,
                max_new_tokens=100,
                # use_cache=True
            )
        # print(answer)
        out_text = llama_tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]
        # print("original out:{}".format(out_text))
        # match = re.search(r'\{(.*?)\}', out_text)
        match = re.search(r'{\s*(\d+)\s*}', out_text)
        
        # 检查是否有匹配结果
        if match:
            extracted_text = int(match.group(1))
            # print(extracted_text)
        # matches = re.findall(r'\{.*?\}', out_text)
        # for match in matches:
        #     result_dict = ast.literal_eval(match)
        
        # print("llama3 answer:{}".format(result_dict['llama_pred']))
        # if debug:
        # print("llama3 prediction:{}".format(extracted_text))
            
        sample_set['correct_answer'] = extracted_text
        sample_set['caption'] = caption
        ans_file.write(json.dumps(sample_set) + "\n")
        ans_file.flush()

    ans_file.close()
    
def run_inference_ego(args):
    """
    Run inference on EgoSchema QA DataSet using our model.

    Args:
        args: Command-line arguments.
    """
    # Initialize the model
    debug = args.debug
    print("Loading LLaMA-VID model !!!")
    model_name = get_model_name_from_path(args.model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(args.model_path, args.model_base, model_name, args.model_max_length)
    print("LLaMA-VID load finish !!")
    print("loading LLaMa3-8B for captioning !!")
    llama_path = args.llama3
    kwargs = {"device_map": "auto"}
    kwargs['torch_dtype'] = torch.float16
    llama_config = LlamaConfig.from_pretrained(llama_path)
    llama_tokenizer = AutoTokenizer.from_pretrained(llama_path, use_fast=True)
    llama_model = LlamaForCausalLM.from_pretrained(llama_path, low_cpu_mem_usage=True, config=llama_config, **kwargs)
    llama_tokenizer.pad_token = llama_tokenizer.eos_token
    print(llama_model.device)
    print("LLaMa3-8B laod finish !!")

    # llama_vid_prompt = [
    #     "What is the background of this movie?",
    #     "What is this movie talking about?",
    #     # "What is the main character of this movie?"
    # ]
    # Load both ground truth file containing questions and answers
    with open(args.gt_file) as file:
        gt_questions = json.load(file)
    gt_questions = get_chunk(gt_questions, args.num_chunks, args.chunk_idx)

    # Create the output directory if it doesn't exist
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    video_formats = ['.mp4', '.avi', '.mov', '.mkv']
    if args.num_chunks > 1:
        output_name = f"{args.num_chunks}_{args.chunk_idx}"
    else:
        output_name = args.output_name
    answers_file = os.path.join(args.output_dir, f"{output_name}.json")
    ans_file = open(answers_file, "w")

    for sample in tqdm(gt_questions, desc="LLaMA-VID with LLaMA-3 in EgoSchema"):
        # video_name = sample['video_id']
        # question = sample['question']
        # id = sample['id']
        # answer = sample['answer']
        video_name = sample['q_uid']
        question = sample['question']
        options = [sample['option 0'], sample['option 1'], sample['option 2'], sample['option 3'], sample['option 4']] 
        # answer = sample['answer']
        # id = sample['question_id']

        sample_set = {'id': video_name, 'question': question, 'options': options}

        # Load the video file
        for fmt in video_formats:  # Added this line
            temp_path = os.path.join(args.video_dir, f"{video_name}.mp4")
            if os.path.exists(temp_path):
                video_path = temp_path
                break

        # Check if the video exists
        if os.path.exists(video_path):
            video = load_video(video_path)
            video = image_processor.preprocess(video, return_tensors='pt')['pixel_values'].half().cuda()
            video = [video]

        # try:
            # Run inference on the video and add the output to the list
        captions = []
        # llama_vid_prompt.append(question)
        llama_vid_prompt = [
        # "What is the background of this movie?",
        "This is a first-person perspective video. Can you describe what the person in the video is doing?",
        "Based on this first-person perspective video, what do you think is the purpose of this person’s behavior in the video?"
        # "What is this video talking about?",
        # "What is the main character of this movie?"
        ]
        llama_vid_prompt.append(question)
        assert len(llama_vid_prompt) == 3
        for pe_prompt in llama_vid_prompt:
            qs = pe_prompt
            if model.config.mm_use_im_start_end:
                qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
            else:
                qs = DEFAULT_IMAGE_TOKEN + '\n' + qs

            conv = conv_templates[args.conv_mode].copy()
            conv.append_message(conv.roles[0], qs)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()

            input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()

            stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
            keywords = [stop_str]
            stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)

            cur_prompt = pe_prompt
            with torch.inference_mode():
                model.update_prompt([[cur_prompt]])
                output_ids = model.generate(
                    input_ids,
                    images=video,
                    do_sample=True,
                    temperature=0.1,
                    max_new_tokens=1024,
                    use_cache=True,
                    stopping_criteria=[stopping_criteria])

            input_token_len = input_ids.shape[1]
            n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
            if n_diff_input_output > 0:
                print(f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')
            outputs = tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0]
            outputs = outputs.strip()
            if outputs.endswith(stop_str):
                outputs = outputs[:-len(stop_str)]
            outputs = outputs.strip()
            captions.append(outputs)
            # print("llama-vid:{}".format(outputs))
            
        caption = prepare_caption(captions)
        # print("llama-vid caption:{}".format(caption))
        llama3_prompt =[
                    {
                        "role": "system",
                        "content":
                            "You are an intelligent chatbot designed for generating safe, complete answers based on instructions provided by user."
                            # "These information come from another multi-modal chatbot, which needs to watch a video and get corresponding answers based on prompts with different attributes."
                    },
                    {
                        "role": "user",
                        "content":
                            "The video-based question and captin pair:\n"
                            f"Question: {question}\n"
                            f"Captions: {caption}\n"
                            "Here are the five answer options for the question:"
                            f"option 0:{options[0]}; option 1:{options[1]}; option 2:{options[2]}; option 3:{options[3]}; option 4:{options[4]}."
                            "Please choose the answer you think is correct among the above options in the form of a Python dictionary string with no keys, where the generated string is between '{' and '}'."
                            "DO NOT PROVIDE ANY OTHER OUTPUT TEXT OR EXPLANATION. You just need to generate the number corresponding to the correct option. "
                            "For example, if you think the right answer is option 0 and your response should look like this: { 0 }."
                    }
                ]
        
        complete_message, message_ids = prepare_prompt(llama_tokenizer, llama3_prompt)
        # print("llama3_messgae:{}".format(complete_message))
        
        ids = torch.tensor(message_ids['input_ids'], dtype=torch.long).to(llama_model.device)
        attention_mask = torch.tensor(message_ids['attention_mask']).to(llama_model.device).unsqueeze(0)
        
        embeddings = llama_model.model.embed_tokens(ids).unsqueeze(0).to(dtype=torch.float16)
        
        with torch.inference_mode():
            output_ids = llama_model.generate(
                inputs_embeds = embeddings,
                attention_mask = attention_mask,
                pad_token_id=llama_tokenizer.eos_token_id,
                do_sample=True,
                temperature=0.1,
                # top_p=args.top_p,
                max_new_tokens=100,
                # use_cache=True
            )
        # print(answer)
        out_text = llama_tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]
        print("original out:{}".format(out_text))
        # match = re.search(r'\{(.*?)\}', out_text)
        match = re.search(r'{\s*(\d+)\s*}', out_text) # etract number from {}

        # 检查是否有匹配结果
        if match:
            extracted_text = int(match.group(1))
            # print(extracted_text)
        # matches = re.findall(r'\{.*?\}', out_text)
        # for match in matches:
        #     result_dict = ast.literal_eval(match)
        
        # print("llama3 answer:{}".format(result_dict['llama_pred']))
        # if debug:
        # print("llama3 prediction:{}".format(extracted_text))
            
        sample_set['correct_answer'] = extracted_text
        sample_set['caption'] = caption
        ans_file.write(json.dumps(sample_set) + "\n")
        ans_file.flush()

    ans_file.close()

def run_inference(args):
    """
    Run inference on ActivityNet QA DataSet using the Video-ChatGPT model.

    Args:
        args: Command-line arguments.
    """
    # Initialize the model
    debug = args.debug
    print("Loading LLaMA-VID model !!!")
    model_name = get_model_name_from_path(args.model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(args.model_path, args.model_base, model_name, args.model_max_length)
    print("LLaMA-VID load finish !!")
    print("loading LLaMa3-8B for captioning !!")
    llama_path = args.llama3
    kwargs = {"device_map": "auto"}
    kwargs['torch_dtype'] = torch.float16
    llama_config = LlamaConfig.from_pretrained(llama_path)
    llama_tokenizer = AutoTokenizer.from_pretrained(llama_path, use_fast=True)
    llama_model = LlamaForCausalLM.from_pretrained(llama_path, low_cpu_mem_usage=True, config=llama_config, **kwargs)
    llama_tokenizer.pad_token = llama_tokenizer.eos_token
    print(llama_model.device)
    print("LLaMa3-8B laod finish !!")

    # llama_vid_prompt = [
    #     "What is the background of this movie?",
    #     "What is this movie talking about?",
    #     # "What is the main character of this movie?"
    # ]
    # Load both ground truth file containing questions and answers
    with open(args.gt_file) as file:
        gt_questions = json.load(file)
    gt_questions = get_chunk(gt_questions, args.num_chunks, args.chunk_idx)

    # Create the output directory if it doesn't exist
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    video_formats = ['.mp4', '.avi', '.mov', '.mkv']
    if args.num_chunks > 1:
        output_name = f"{args.num_chunks}_{args.chunk_idx}"
    else:
        output_name = args.output_name
    answers_file = os.path.join(args.output_dir, f"{output_name}.json")
    ans_file = open(answers_file, "w")

    for sample in tqdm(gt_questions, desc="LLaMA-VID with LLaMA-3 in {}:".format(args.data_set)):
        # video_name = sample['video_id']
        # question = sample['question']
        # id = sample['id']
        # answer = sample['answer']
        video_name = sample['video']
        question = sample['question']
        answer = sample['answer']
        id = sample['question_id']

        sample_set = {'id': id, 'question': question, 'answer': answer}

        # Load the video file
        for fmt in video_formats:  # Added this line
            temp_path = os.path.join(args.video_dir, f"{video_name}")
            if os.path.exists(temp_path):
                video_path = temp_path
                break

        # Check if the video exists
        if os.path.exists(video_path):
            video = load_video(video_path)
            video = image_processor.preprocess(video, return_tensors='pt')['pixel_values'].half().cuda()
            video = [video]

        # try:
            # Run inference on the video and add the output to the list
        captions = []
        # llama_vid_prompt.append(question)
        llama_vid_prompt = [
        # "What is the background of this movie?",
        "What can you see from this video? What's the relationship between them?",
        "Describe this video as much detail as possible."
        # "What is this video talking about?",
        # "What is the main character of this movie?"
        ]
        llama_vid_prompt.append(question)
        assert len(llama_vid_prompt) == 3
        for pe_prompt in llama_vid_prompt:
            qs = pe_prompt
            if model.config.mm_use_im_start_end:
                qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
            else:
                qs = DEFAULT_IMAGE_TOKEN + '\n' + qs

            conv = conv_templates[args.conv_mode].copy()
            conv.append_message(conv.roles[0], qs)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()

            input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()

            stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
            keywords = [stop_str]
            stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)

            cur_prompt = pe_prompt
            with torch.inference_mode():
                model.update_prompt([[cur_prompt]])
                output_ids = model.generate(
                    input_ids,
                    images=video,
                    do_sample=True,
                    temperature=0.1,
                    max_new_tokens=1024,
                    use_cache=True,
                    stopping_criteria=[stopping_criteria])

            input_token_len = input_ids.shape[1]
            n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
            if n_diff_input_output > 0:
                print(f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')
            outputs = tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0]
            outputs = outputs.strip()
            if outputs.endswith(stop_str):
                outputs = outputs[:-len(stop_str)]
            outputs = outputs.strip()
            captions.append(outputs)
            # print("llama-vid:{}".format(outputs))
            
        caption = prepare_caption(captions)
        # print("llama-vid caption:{}".format(caption))
        llama3_prompt =[
                    {
                        "role": "system",
                        "content":
                            "You are an intelligent chatbot designed for generating safe, complete answers based on instructions provided by user."
                            # "These information come from another multi-modal chatbot, which needs to watch a video and get corresponding answers based on prompts with different attributes."
                    },
                    {
                        "role": "user",
                        "content":
                            "The video-based question and captin pair:\n"
                            f"Question: {question}\n"
                            f"Captions: {caption}\n"
                            "Please generate the answer of the question based on the information provided."
                            "Please generate the response in the form of a Python dictionary string with no keys, where the generated string is between '{' and '}'."
                            "DO NOT PROVIDE ANY OTHER OUTPUT TEXT OR EXPLANATION. Please make supplementary summaries in order to provide a richer and more comprehensive answer. Only provide the right answer string. "
                            "For example, your response should look like this: { YOUR ANSWER }."
                            # "Please generate the response in the form of a Python dictionary string with keys 'llama_pred', where value of 'llama_pred' is the strings generated."
                            # "DO NOT PROVIDE ANY OTHER OUTPUT TEXT OR EXPLANATION. Only provide the right answer string. "
                            # "For example, your response should look like this: {'llama_pred': 'YOUR ANSWER' }."
                    }
                ]
        
        complete_message, message_ids = prepare_prompt(llama_tokenizer, llama3_prompt)
        # print("llama3_messgae:{}".format(complete_message))
        
        ids = torch.tensor(message_ids['input_ids'], dtype=torch.long).to(llama_model.device)
        attention_mask = torch.tensor(message_ids['attention_mask']).to(llama_model.device).unsqueeze(0)
        
        embeddings = llama_model.model.embed_tokens(ids).unsqueeze(0).to(dtype=torch.float16)
        
        with torch.inference_mode():
            output_ids = llama_model.generate(
                inputs_embeds = embeddings,
                attention_mask = attention_mask,
                pad_token_id=llama_tokenizer.eos_token_id,
                do_sample=True,
                temperature=0.1,
                # top_p=args.top_p,
                # max_new_tokens=512,
                # use_cache=True
            )
        # print(answer)
        out_text = llama_tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]
        # print("original out:{}".format(out_text))
        match = re.search(r'\{(.*?)\}', out_text)

        # 检查是否有匹配结果
        if match:
            extracted_text = match.group(1)
        # matches = re.findall(r'\{.*?\}', out_text)
        # for match in matches:
        #     result_dict = ast.literal_eval(match)
        
        # print("llama3 answer:{}".format(result_dict['llama_pred']))
        # if debug:
        # print("llama3 prediction:{}".format(extracted_text))
            
        sample_set['pred'] = extracted_text
        sample_set['caption'] = caption
        ans_file.write(json.dumps(sample_set) + "\n")
        ans_file.flush()

    ans_file.close()


if __name__ == "__main__":
    args = parse_args()
    if args.data_set in ['msvd', 'msrvtt']:
        run_inference(args)
    elif args.data_set in ['egoschema']:
        run_inference_ego(args)
    elif args.data_set in ['next-qa']:
        run_inference_next(args)
    elif args.data_set in ['intent-qa']:
        run_inference_intent(args)
