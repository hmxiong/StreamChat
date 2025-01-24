import math
import os
import argparse
import json
import torchaudio
import numpy as np

from tqdm import tqdm
from llava.eval.model_utils import load_video
from decord import VideoReader, cpu

from llavanext.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llavanext.conversation import conv_templates, SeparatorStyle
from llavanext.model.builder import load_pretrained_model
from llavanext.utils import disable_torch_init
from llavanext.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path
from utiles import compute_gradients, Optical_flow, SSIM, long_short_memory_update, search_tree, TreeNode, build_prompt_with_search_memory_only_related, convert_to_markdown, visualize_memory_feature_with_PCA
# from llama_index.legacy.llms import (HuggingFaceLLM, CustomLLM, CompletionResponse, CompletionResponseGen, LLMMetadata)
# from llama_index.core.llms.callbacks import llm_completion_callback
from typing import Optional, List, Mapping, Any
from memory_bank.memory_retrieval.local_doc_qa import LocalMemoryRetrieval
from memory_bank.memory_utils import summarize_memory_event_personality, enter_name, save_local_memory
from memory_bank.summarize_memory import LLMClientLLaMA3
from transformers import EvalPrediction, Trainer
from memory_bank.prompt_utils import *

import torch.nn.functional as F

from PIL import Image
import math
import torch
import time
import cv2
import threading
import queue
import time


# Global variables for shared resources
# frame_bank = queue.Queue()
# feature_bank = queue.Queue()
frame_bank   = []
feature_bank = []
short_memory_buffer = []
time_count = 0
time_index = 0
time_triger = False
finish_triger = True
long_memory_tree= TreeNode
condition = threading.Condition()
mutex = threading.Lock()

question_list = [
        # {
        #     "question": "This is a clip from the movie 'Truman'. Please describe the appearance of the ship in this video clip.",
        #     "answer": "Truman's boat is a small white sailboat with wooden masts and white sails.",
        #     "class": "describe",
        #     "time": 240
        # },
        {
            "question": "What happened to the boat shown in the picture?",
            "answer": "The boat hit a wall while sailing, making it impossible to continue moving forward.",
            "class": "describe",
            "time": 1320
        },
        {
            "question": "Why does Truman keep hitting the wall?",
            "answer":"The boat he was driving crashed into a wall, causing him to become stranded at sea, leaving him furious and frustrated.",
            "class": "describe",
            "time": 1440
        },
        {
            "question": "How do Truman feel and what's the reason ?",
            "answer":"Truman feels very frustrated. From the previous information, I learned that his boat was stuck on the sea and could not continue sailing.",
            "class": "describe",
            "time": 2232
        },
        {
            "question": "What is Truman wearing black shirt doing?",
            "answer":"Truman seemed to have found a way out, and he was following it.",
            "class": "describe",
            "time": 3120
        },
        {
            "question": "Why does Truman climb the steps?",
            "answer":"Before, he was stuck at sea, now he found a way to the exit door, so he climbed the steps towards that door.",
            "class": "describe",
            "time": 3600
        },
        {
            "question": "Now what are the characteristics of this man you see, is he the man trapped at sea? ",
            "answer":"The man wears a brown hat and glasses, but he is not a man trapped on the sea.",
            "class": "describe",
            "time": 5280
        },
        {
            "question": "According to the scene you saw, did Truman trapped on the sea find a way out?",
            "answer":"Yes, he find the door to leave, and he is standing in front of that door.",
            "class": "describe",
            "time": 6000
        },
        {
            "question": "What are people in bar doing?",
            "answer":"These people were watching TV, and what was playing on the TV was Truman's live broadcast.",
            "class": "describe",
            "time": 7608
        },
        {
            "question": "Why is this man watching Truman Live while taking a bath in the bathtub so happy?",
            "answer":"Because he is also happy that Truman find a way to leave.",
            "class": "describe",
            "time": 8880
        },
        {
            "question": "How many women are there on the table watching Truman Live?",
            "answer":"There are two women。",
            "class": "describe",
            "time": 9120
        }
    ]
    
time_line = [int(ques['time']) for ques in question_list]
    
def parse_args():
    """
    Parse command-line arguments.
    """
    parser = argparse.ArgumentParser()

    # Define the command-line arguments
    parser.add_argument('--video_dir', help='Directory containing video files.', required=True)
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--conv-mode", type=str, required=False, default='video-chatgpt_v1')
    parser.add_argument("--mode", type=str, required=False, default='off_line')
    parser.add_argument("--num_chunks", type=int, default=1)
    parser.add_argument("--chunk_idx", type=int, default=0)
    parser.add_argument("--num_frames", type=int, default=4)
    parser.add_argument("--device", type=str, required=False, default='cuda:0')
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--memory_basic_dir", type=str, required=True, default='/13390024681/llama/EfficientVideo/Ours/memory_bank/memories')
    parser.add_argument("--memory_file", type=str, required=True, default='updata_memories_for_streaming.json')
    parser.add_argument("--language", type=str, required=True, default='en')
    parser.add_argument("--memory_search_top_k", type=int, default=1)
    parser.add_argument("--ppl", action="store_true", help="weather to calculat ppl")
    

    return parser.parse_args()

def llava_inference_with_embedding(question, num_frames, conv_mode, model, tokenizer, chat, short_memory_buffer_cache, long_memory_tree_cache, history_prompt=None):
    
    global feature_bank
    
    if history_prompt is not None:
        if model.config.mm_use_im_start_end:
            qs = history_prompt + DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + question
        else:
            qs = history_prompt + DEFAULT_IMAGE_TOKEN + '\n' + question
    else:
        if model.config.mm_use_im_start_end:
            qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + question
        else:
            qs = DEFAULT_IMAGE_TOKEN + '\n' + question

    print("Question:{}".format(qs))
    
    conv = conv_templates[conv_mode].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()
    # print(question)
    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt')
    input_ids = input_ids.unsqueeze(0).cuda()
    # ques_ids = ques_ids.unsqueeze(0).cuda()
    # print("input",input_ids)
    # print("ques",ques_ids)
    # print("image size 1 :{}".format(image_sizes))
    short_memory_embedding = torch.cat(short_memory_buffer_cache).view(-1, short_memory_buffer_cache[0].shape[-1]) # [4x576, 4096]
    
    time_0 = time.time()
    # with threading.Lock():
    # image_embeddings, topk_indices, topk_values = topk_feature(feature_list, num_frames, model, question, tokenizer) # x 576 dimension
    question_ids = tokenizer(question).input_ids
    question_embeddings  = model.get_model().embed_tokens(torch.tensor(question_ids, dtype=torch.long, device='cuda')) # num_text_token 4096
    if long_memory_tree_cache is not None:
        long_memory_list = search_tree(long_memory_tree_cache, torch.cat([question_embeddings, short_memory_embedding], dim=0))
        long_memory_embeddings = torch.cat(long_memory_list, dim=0).view(-1, long_memory_list[0].shape[-1]) # [40x36, 4096]
        
        image_embeddings = torch.cat([short_memory_embedding, long_memory_embeddings], dim=0)
        visualize_memory_feature_with_PCA(feature_bank, long_memory_list, clustering=5, same_color=False, only_most_important=False)
        
    else:
        image_embeddings = short_memory_embedding
    
    
    time_1 = time.time()
    # print("image embedding shape:{}".format(image_embeddings.shape))
    # topk_images = [frame_list[idx] for idx in topk_indices]
    # for index, image in enumerate(topk_images):
    #     img = Image.fromarray(image)
    #     img.save("/13390024681/llama/EfficientVideo/Ours/save_images/topk_{}.jpg".format(topk_values[index]))
    
    with torch.inference_mode():
        output_ids = model.generate_with_image_embedding(
            input_ids,
            image_embeddings=[image_embeddings],
            # question_ids=ques_ids,
            # modalities="image",
            do_sample=True if args.temperature > 0 else False,
            temperature=args.temperature,
            top_p=args.top_p,
            num_beams=args.num_beams,
            max_new_tokens=256,
            use_cache=False)

    outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
    time_2 = time.time()
    # Perform inference and play the generated audio
    # wavs = chat.infer([outputs])
    # Audio(wavs[0], rate=24_000, autoplay=True)
    # Save the generated audio 
    # torchaudio.save("output.wav", torch.from_numpy(wavs[0]), 24000)
    if chat is not None:
        chat.tts_to_file(text="{}".format(outputs), 
                    speaker_wav="/13390024681/llama/EfficientVideo/Ours/female.wav", 
                    language="en",
                    file_path="output.wav")
    
    time_3 = time.time() # TTS havy time dely
    
    print("process time:{}, generate time:{}".format((time_1 - time_0), (time_2 - time_1)))
    
    return outputs, (time_1 - time_0), (time_2 - time_1)

def llava_inference_with_embedding_and_ppl(question, label, num_frames, conv_mode, model, tokenizer, chat, short_memory_buffer_cache, long_memory_tree_cache, history_prompt=None):
    
    if history_prompt is not None:
        if model.config.mm_use_im_start_end:
            qs = history_prompt + DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + question + label
        else:
            qs = history_prompt + DEFAULT_IMAGE_TOKEN + '\n' + question + label
    else:
        if model.config.mm_use_im_start_end:
            qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + question + label
        else:
            qs = DEFAULT_IMAGE_TOKEN + '\n' + question + label

    # print("Question:{}".format(qs))
    
    conv = conv_templates[conv_mode].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()
    # print(question)
    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt')
    input_ids = input_ids.unsqueeze(0).cuda()
    label = input_ids
    # ques_ids = ques_ids.unsqueeze(0).cuda()
    # print("input",input_ids)
    # print("ques",ques_ids)
    # print("image size 1 :{}".format(image_sizes))
    short_memory_embedding = torch.cat(short_memory_buffer_cache).view(-1, short_memory_buffer_cache[0].shape[-1]) # [4x576, 4096]
    
    time_0 = time.time()
    # with threading.Lock():
    # image_embeddings, topk_indices, topk_values = topk_feature(feature_list, num_frames, model, question, tokenizer) # x 576 dimension
    question_ids = tokenizer(question).input_ids
    question_embeddings  = model.get_model().embed_tokens(torch.tensor(question_ids, dtype=torch.long, device='cuda')) # num_text_token 4096
    if long_memory_tree_cache is not None:
        long_memory_list = search_tree(long_memory_tree_cache, torch.cat([question_embeddings, short_memory_embedding], dim=0))
        long_memory_embeddings = torch.cat(long_memory_list, dim=0).view(-1, long_memory_list[0].shape[-1]) # [40x36, 4096]
        
        image_embeddings = torch.cat([short_memory_embedding, long_memory_embeddings], dim=0)
    else:
        image_embeddings = short_memory_embedding
    time_1 = time.time()
    # print("image embedding shape:{}".format(image_embeddings.shape))
    # topk_images = [frame_list[idx] for idx in topk_indices]
    # for index, image in enumerate(topk_images):
    #     img = Image.fromarray(image)
    #     img.save("/13390024681/llama/EfficientVideo/Ours/save_images/topk_{}.jpg".format(topk_values[index]))
    
    # with torch.inference_mode():
    with torch.no_grad():
        output = model.forward_with_fix_embedding(
                input_ids,
                attention_mask = None,
                position_ids = None,
                past_key_values = None,
                inputs_embeds = None,
                labels = label,
                images = [image_embeddings],
                image_sizes=None,
                return_dict = True)

    time_2 = time.time()
    # Perform inference and play the generated audio
    # wavs = chat.infer([outputs])
    # Audio(wavs[0], rate=24_000, autoplay=True)
    # Save the generated audio 
    # torchaudio.save("output.wav", torch.from_numpy(wavs[0]), 24000)
    
    
    time_3 = time.time() # TTS havy time dely
    
    print("process time:{}, generate time:{}".format((time_1 - time_0), (time_2 - time_1)))
    
    return output, (time_1 - time_0), (time_2 - time_1)

def updating_memory_buffer(update_event, start_inference_event):
    """
    Thread function to handle user input.
    """
    global short_memory_buffer
    global long_memory_tree
    global feature_bank
    global time_triger
    
    while True:
        # if not start_inference_triger and finish_infernece:
        
        # if len(feature_bank) > 20:
        if time_triger:
            update_event.wait()
            print("<<<< start building memory >>>>")
            cache = feature_bank
            if len(cache) > 8:
                short_memory_buffer, long_memory_tree = long_short_memory_update(cache, short_window=8, 
                                                                                remember_window=4, tau=10, 
                                                                                compress_rate=2, chunk_size=50, 
                                                                                num_clusters=5, interval=2)
                print("<<<< memory building finish >>>>")
            else:
                print("<<<< low cache mode not need long memory >>>>")
                short_memory_buffer = [cache[i] for i in range(len(cache))]
                long_memory_tree = None
                print("short_memory_buffer", len(short_memory_buffer))
            
            assert len(short_memory_buffer) > 0, "No memory ?"
            
            update_event.clear()
            start_inference_event.set()
        else:
            time.sleep(0.5)
        # else:
        #     time.sleep(0.5) # avoid GIL

def user_input_thread(input_queue, pause_event):
    """
    Thread function to handle user input.
    """
    while True:
        pause_event.wait()
        question = input("User Instruction: ")
        input_queue.put(question)
        if question.lower() == 'exit':
            break
        # Pause the user input thread
        pause_event.clear()

def topk_feature(feature_list, num_frm, model, question_text, tokenizer):
    """
    Function to process a single video frame and prepare it for inference.
    """
    # frame bank contains all the video frame that you need 
    # Placeholder function for demonstration purposes.
    # In actual implementation, adapt this function to your needs.
    # total_frames = len(feature_bank)
    # feature_bank = queue_to_list(feature_bank)
    total_frames = len(feature_list)
    question_ids = tokenizer(question_text).input_ids
    question_embeddings  = model.get_model().embed_tokens(torch.tensor(question_ids, dtype=torch.long, device='cuda'))
    # print("question embeddings : {}".format(question_embeddings.shape)) # num_text_token 4096
    # print("There are total :{} frames in this video.".format(total_frames)) # how to speed up ?

    all_image_features = torch.cat(feature_list) # 1 576 4096
    # print("all_image_features shape :{}".format(all_image_features.shape))
    # time_2 = time.time()
    simarity = all_image_features @ question_embeddings.permute(1, 0) # num_frame 576 num_text_token
    simarity = simarity.sum(dim=1).sum(dim=1)
    # time_3 = time.time()
    # print("simarity shape :{}".format(simarity.shape))
    topk_values, topk_indices = torch.topk(simarity, num_frm)
    topk_features = [feature_list[idx] for idx in topk_indices]
    # Concatenate all top-k features
    concatenated_features = torch.cat(topk_features, dim=0) # num_select 576 4096
    concatenated_features = concatenated_features.view(-1, concatenated_features.shape[-1])
    # print("concatenated_features shape :{}".format(concatenated_features.shape))

    return concatenated_features, topk_indices, topk_values
 
def video_reader_thread_with_embedding(cap, total_frames, frame_rate, image_processor, model, count):
    """
    Thread function to read video frames and put them into a queue.
    """
    # 两个线程之间发生一定的资源互占的情况
    
    global frame_bank
    global feature_bank
    global time_count
    global time_triger
    global time_index
    
    current_frame_rate = 0
    # count = 0
    last_frame = None
    change_time = 0
    
    time_bank_1 = []
    time_bank_2 = []
    all_time_bank = []
    mag_bank = []
    
    pbar = tqdm(total=total_frames, desc="Processing Video", unit="frame")
    
    while cap.isOpened() and current_frame_rate < total_frames:
        time_1 = time.time()
        ret, frame = cap.read()
        if not ret:
            break
        current_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # Save the last frame to image_show.jpg
        if time_count > 1:
            time_2 = time.time()
            # is_change, mean_mag, current_frame_tensor = SSIM(last_frame, current_frame, image_processor, model, 0.9) # judging by SSIM
            is_change, mean_mag, current_frame_tensor = Optical_flow(last_frame, current_frame, image_processor, model, 0.18) # judging by optical flow
            torch.cuda.empty_cache()
            time_3 = time.time()
            mag_bank.append(mean_mag)
            all_time_bank.append((time_3-time_2))
            if is_change:
                change_time += 1
                time_4 = time.time()
                image_embedding = model.only_encode(current_frame_tensor.unsqueeze(0).to(dtype=torch.float16))
                # print("image embedding without proj:{}".format(image_embedding.shape)) # 1 576 1024
                image_embedding = model.only_project(image_embedding)
                time_5 = time.time()
                # 确保有足够的帧数进行推理
                with mutex:
                    # Enqueue frame and feature
                    # frame_bank.put(current_frame)
                    # feature_bank.put(image_embedding)
                    # frame_bank.append(current_frame)
                    feature_bank.append(image_embedding)
                time_6 = time.time()
                img = Image.fromarray(current_frame)
                img.save("/13390024681/llama/EfficientVideo/Ours/image_show.jpg")
                # print("time spend analyze:{}/{}/{}/{}/{}".format((time_6 - time_5), (time_5 - time_4), (time_4 - time_3), (time_3 - time_2), (time_2 - time_1)))
                # time_bank_1.append((time_5 - time_4))
                # time_bank_2.append((time_3 - time_2))
                # all_time_bank.append((time_5 - time_4))
                # if len(time_bank_1) > 100 and len(time_bank_2) > 100:
                #     total_time_1 = sum(time_bank_1)
                #     total_time_2 = sum(time_bank_2)
                #     # count = len(time_bank_1)
                #     average_time_1 = total_time_1 / len(time_bank_1)
                #     average_time_2 = total_time_2 / len(time_bank_1)
                #     print("avg time 1:{}/ avg time 2:{}".format(average_time_1, average_time_2))
                #     print("count:{}".format(count))
                #     print("max mag {} and min mag {}".format(max(mag_bank), min(mag_bank)))
                #     assert 1==2
            elif time_count == 2: # 保留一个初始帧的特征
                image_embedding = model.only_encode(current_frame_tensor.unsqueeze(0).to(dtype=torch.float16))
                # print("image embedding without proj:{}".format(image_embedding.shape)) # 1 576 1024
                image_embedding = model.only_project(image_embedding)

                feature_bank.append(image_embedding)
            
        else:
            mean_mag = 0.0
            all_time_bank.append(0.00001)
       
       
        last_frame = current_frame
        # current_frame_rate += 1
        # with condition:
        time_count += 1
        if time_count in time_line:
            time_triger = True
            time_index = time_index + 1
        # condition.notify()  # 通知等待的线程
        # time_count += 1
        time_7 = time.time()
        FPS = (time_count)/sum(all_time_bank)
        # print("FPS:{}".format(FPS))
        # Update tqdm progress bar and set postfix for FPS
        pbar.set_postfix(FPS="{:.2f}".format(FPS), MAG="{:.2f}".format(mean_mag), Time="{}".format(time_count), Buffer="{}".format(len(feature_bank)))
        pbar.update(1)
        
    cap.release()
    print("Video processing completed.")
    print("Find chanement {} times in {}".format(change_time, total_frames))
    
def inference_thread_with_emebdding_test(input_queue, model, tokenizer, image_processor, num_frames, conv_mode, pause_event, update_event, chat):
    """
    Thread function to run inference on video frames.
    """
    global frame_bank
    global feature_bank
    global short_memory_buffer
    global long_memory_tree
    
    question_list = [
        "please tell me what you see from these images?",
        "please tell me what you see from these images?",
        "please tell me what you see from these images?",
        "please tell me what you see from these images?",
        "please tell me what you see from these images?",
        "please tell me what you see from these images?",
        "please tell me what you see from these images?",
        "please tell me what you see from these images?",
        "please tell me what you see from these images?",
        "please tell me what you see from these images?",
        "please tell me what you see from these images?"
    ]
    
    process_time_bank = []
    generate_time_bank = []
    time_step = []
    count = 0
    
    while True:
        # if not input_queue.empty():
        # if frame_bank.qsize() % 15 == 0 and frame_bank.qsize() != 0:
        # if len(frame_bank) % 40 == 0 and len(frame_bank) != 0:
        if update_event.wait():
            short_memory_buffer_cache = short_memory_buffer
            long_memory_tree_cache = long_memory_tree
            print("<<<< transfer finish and start to inference >>>>")
            question = question_list[count]
            count += 1
            if question.lower() == 'exit':
                print("Exit command received. Terminating inference.")
                break
            
        
            # Ensure enough frames are available for inference
            with mutex:
                # if feature_bank.qsize() < num_frames:
                if len(feature_bank) < num_frames:
                    print("Not enough frames for inference. Waiting for more frames...")
                    # time.sleep(1)
                    continue
            
            # Retrieve frames and features from queues
            # with mutex:
            #     frames = [frame_bank.get() for _ in range(feature_bank.qsize())]
            #     features = [feature_bank.get() for _ in range(feature_bank.qsize())]
            length= len(feature_bank)
            print("feature length :{}".format(length))
            # print("frame length :{}".format(len(frame_bank)))
            
            output, process_time, generate_time = llava_inference_with_embedding(question, num_frames, 
                                                                                 conv_mode, model,
                                                                                 tokenizer, chat, 
                                                                                 short_memory_buffer_cache, long_memory_tree_cache)
            
            process_time_bank.append(process_time)
            generate_time_bank.append(generate_time)
            time_step.append(length)
            
            print("LLaVA:", output)
            pause_event.set()
            if len(generate_time_bank) == len(question_list):
                total_time_1 = sum(process_time_bank)
                total_time_2 = sum(generate_time_bank)
                count = len(generate_time_bank)
                average_time_1 = total_time_1 / count
                average_time_2 = total_time_2 / count
                print("avg process time:{}/ avg generate time:{}".format(average_time_1, average_time_2))
                print("total process time:{}".format(process_time_bank))
                print("total generate time:{}".format(generate_time_bank))
                print("total time step:{}".format(time_step))
                assert 1==2
            # frame_bank   = []  # Clear frame bank for next inference
        else:
            time.sleep(0.5)

def inference_thread_with_memory_test(start_inference_event, model, tokenizer, image_processor, num_frames, conv_mode, pause_event, update_event, chat):
    """
    Thread function to run inference on video frames.
    """
    global frame_bank
    global feature_bank
    global short_memory_buffer
    global long_memory_tree
    
    question_list = [
        "please tell me what you see from these images?",
        "I saw the pilot was performing a mission, what is the time limit for the mission?",
        "Is mavrrick the pilot in the film?",
        "please tell me what you see from this video?",
        "please tell me what you see from this video?",
        "What was everyone's expression like when they saw the pilot completing the mission?",
        "please tell me what you see from this video?",
        "please tell me what you see from this video?",
        "please tell me what you see from this video?",
        "please tell me what you see from this video?",
        "please tell me what you see from this video?"
    ]
    
    process_time_bank = []
    generate_time_bank = []
    time_step = []
    count = 0
    
    while True:
        # if not input_queue.empty():
        # if frame_bank.qsize() % 15 == 0 and frame_bank.qsize() != 0:
        if len(frame_bank) % 40 == 0 and len(frame_bank) != 0:
        # if update_event.wait():
        # if start_inference_triger:
            start_inference_event.wait()
            # start_inference_triger = False
            # finish_infernece = False
            short_memory_buffer_cache = short_memory_buffer
            long_memory_tree_cache = long_memory_tree
            print("<<<< transfer finish and start to inference >>>>")
            question = question_list[count]
            count += 1
            if question.lower() == 'exit':
                print("Exit command received. Terminating inference.")
                break
            
        
            # Ensure enough frames are available for inference
            # with mutex:
            #     # if feature_bank.qsize() < num_frames:
            #     if len(feature_bank) < num_frames:
            #         print("Not enough frames for inference. Waiting for more frames...")
            #         # time.sleep(1)
            #         continue
            
            # Retrieve frames and features from queues
            # with mutex:
            #     frames = [frame_bank.get() for _ in range(feature_bank.qsize())]
            #     features = [feature_bank.get() for _ in range(feature_bank.qsize())]
            # length= len(feature_bank)
            # print("feature length :{}".format(length))
            # print("frame length :{}".format(len(frame_bank)))
            
            output, process_time, generate_time = llava_inference_with_embedding(question, num_frames, 
                                                                                 conv_mode, model,
                                                                                 tokenizer, chat, 
                                                                                 short_memory_buffer_cache, long_memory_tree_cache)
            
            process_time_bank.append(process_time)
            generate_time_bank.append(generate_time)
            # time_step.append(length)
            
            print("LLaVA:", output)
            pause_event.set()
            start_inference_event.clear()
            update_event.set()
            finish_infernece = True
            if len(generate_time_bank) == len(question_list):
                total_time_1 = sum(process_time_bank)
                total_time_2 = sum(generate_time_bank)
                count = len(generate_time_bank)
                average_time_1 = total_time_1 / count
                average_time_2 = total_time_2 / count
                print("avg process time:{}/ avg generate time:{}".format(average_time_1, average_time_2))
                print("total process time:{}".format(process_time_bank))
                print("total generate time:{}".format(generate_time_bank))
                # print("total time step:{}".format(time_step))
                assert 1==2
            # frame_bank   = []  # Clear frame bank for next inference
        else:
            time.sleep(0.5)
            
def inference_thread_with_emebdding(input_queue, model, tokenizer, image_processor, num_frames, conv_mode, pause_event, chat):
    """
    Thread function to run inference on video frames.
    """
    global frame_bank
    global feature_bank
    
    process_time_bank = []
    generate_time_bank = []
    
    count = 0
    while True:
        if not input_queue.empty():
        # if frame_bank.qsize() // 15 == 0 and frame_bank.qsize() != 0:
            question = input_queue.get()
            if question.lower() == 'exit':
                print("Exit command received. Terminating inference.")
                break
            
        
            # Ensure enough frames are available for inference
            with mutex:
                if feature_bank.qsize() < num_frames:
                    print("Not enough frames for inference. Waiting for more frames...")
                    # time.sleep(1)
                    continue
            
            # Retrieve frames and features from queues
            with mutex:
                frames = [frame_bank.get() for _ in range(num_frames)]
                features = [feature_bank.get() for _ in range(num_frames)]

            output, process_time, generate_time = llava_inference_with_embedding(question, conv_mode, model,
                                    tokenizer, chat, features, frames)
            
            # process_time_bank.append(process_time)
            # generate_time_bank.append(generate_time)
            print("LLaVA:", output)
            pause_event.set()
            # frame_bank   = []  # Clear frame bank for next inference
        time.sleep(0.5)
        
def inference_thread_with_memory_and_dialogue_retrival_test(start_inference_event, model, tokenizer, image_processor, num_frames, conv_mode, pause_event, update_event, chat, memory_config, args, current_frame, output_loss=False):
    """
    Thread function to run inference on video frames.
    """
    global frame_bank
    global feature_bank
    global short_memory_buffer
    global long_memory_tree
    global time_count
    global time_triger
    global finish_triger
    global time_index
    
    
    print("time_line", time_line)
    process_time_bank = []
    generate_time_bank = []
    time_step = []
    all_ppl = []
    avg_ppl = []
    count = 0
        
    user_memory = memory_config['user_memory']
    user_name = memory_config['user_name']
    user_memory_index = memory_config['user_memory_index']
    local_memory_qa = memory_config['local_memory_qa']
    only_related_prompt = memory_config['only_related_prompt']
    # new_user_meta_prompt = memory_config['new_user_meta_prompt']
    user_keyword = memory_config['user_keyword']
    ai_keyword = memory_config['ai_keyword']
    boot_actual_name = memory_config['boot_actual_name']
    memory = memory_config['memory']
    
    while True:
        # print()
        # if not input_queue.empty():
        # if frame_bank.qsize() % 15 == 0 and frame_bank.qsize() != 0:
        # if len(feature_bank) % 40 == 0 and len(feature_bank) != 0:
        # with condition:
        #     condition.wait()  # 等待x值的更新通知
        #     if x in list_data:
        if time_triger and finish_triger:
            index = time_index - 1
        # if update_event.wait():
        # if start_inference_triger:
            start_inference_event.wait()
            # start_inference_triger = False
            # finish_infernece = False
            short_memory_buffer_cache = short_memory_buffer
            long_memory_tree_cache = long_memory_tree
            print("<<<< transfer finish and start to inference >>>>")
            question = question_list[index]['question']
            labels = question_list[index]['answer']
            count += 1
            if question.lower() == 'exit':
                print("Exit command received. Terminating inference.")
                break
            # print("<<<< transfer finish and start to inference >>>>")
            
            print("<<<< Retrival context >>>>")
            searched_history = build_prompt_with_search_memory_only_related(question, user_name, user_memory_index, local_memory_qa, only_related_prompt, user_keyword, ai_keyword, boot_actual_name)
            # print("searched_history:{}".format(searched_history))

            if output_loss:
                output_dict, _, _ = llava_inference_with_embedding_and_ppl(question, labels, num_frames, 
                                                                                 conv_mode, model,
                                                                                 tokenizer, chat, 
                                                                                 short_memory_buffer_cache, long_memory_tree_cache, searched_history)
                loss = output_dict.loss
                # loss = loss.detach().cpu().numpy()
                ppl = torch.exp(loss)
                print("loss:{}/ppl:{}".format(loss, ppl))
                all_ppl.append(ppl.detach().cpu().numpy())
            # else:
            output, process_time, generate_time = llava_inference_with_embedding(question, num_frames, 
                                                                                conv_mode, model,
                                                                                tokenizer, chat, 
                                                                                short_memory_buffer_cache, long_memory_tree_cache, searched_history)
            print("LLaVA:", output)
                
            process_time_bank.append(process_time)
            generate_time_bank.append(generate_time)
            # time_step.append(length)
            
            # print("LLaVA:", output)
            
            # update memory 
            b = [[question, output]]
            # a, b = [[y[0], convert_to_markdown(y[1])] for y in history] ,history 
            if user_name:
                memory = save_local_memory(memory,b,user_name,args)
            
            _,_,memory,user_name,user_memory_index = enter_name(user_name,memory,local_memory_qa,args)
            
            torch.cuda.empty_cache()
            pause_event.set()
            start_inference_event.clear()
            update_event.set()
            # finish_infernece = True
            time_triger = False
            
            # if len(generate_time_bank) == len(question_list):
            #     total_time_1 = sum(process_time_bank)
            #     total_time_2 = sum(generate_time_bank)
            #     count = len(generate_time_bank)
            #     average_time_1 = total_time_1 / count
            #     average_time_2 = total_time_2 / count
            #     print("avg process time:{}/ avg generate time:{}".format(average_time_1, average_time_2))
            #     print("total process time:{}".format(process_time_bank))
            #     print("total generate time:{}".format(generate_time_bank))
            #     # print("total time step:{}".format(time_step))
            #     assert 1==2
            # frame_bank   = []  # Clear frame bank for next inference
        else:
            if len(all_ppl) == len(question_list):
                print("Avg PPL:",sum(all_ppl)/len(all_ppl))
                differences = [abs(all_ppl[i+1] - all_ppl[i]) for i in range(len(all_ppl) - 1)]
                total_difference = sum(differences)
                count_differences = len(differences)
                average_difference = total_difference / count_differences
                print("Fluency PPL:", average_difference)
                
            time.sleep(0.1)
                          
def run_inference(args):
    """
    Run inference on Video QA DataSetå.

    Args:
        args: Command-line arguments.
    """

    disable_torch_init()
    inference_mode = args.mode
    model_path = os.path.expanduser(args.model_name)
    model_name = get_model_name_from_path(model_path)
    print("Initialize GPT-4o in LLaVA-NExT version:{} in {} mode !".format(args.conv_mode, inference_mode))
    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, args.model_base, model_name)
    # print("Initialize TTS for text-to-speech transform !!")
    # Init TTS with the target model name
    # Get device
    # Initialize and load the model: 
    # chat = ChatTTS.Chat()
    # chat.load_models(source='custom',
    #                 custom_path='/13390024681/All_Model_Zoo/ChatTTS',
    #                 compile=True) # Set to True for better performance
    # chat = TTS(model_path="/13390024681/All_Model_Zoo/XTTS-v2",config_path="/13390024681/All_Model_Zoo/XTTS-v2/config.json", progress_bar=False).to(model.device)
    chat = None
    print("ALl model get ready !!")
    
    if args.ppl:
        print("Need to calculate PPL during test !!")
        
    print("Building memory context retriveler !!")
    memory_dir = os.path.join(args.memory_basic_dir,args.memory_file)
    # print(memory_dir)
    if not os.path.exists(memory_dir):
        json.dump({},open(memory_dir,"w",encoding="utf-8"))

    language = args.language
    print("Storing memory in {} using Language:{}".format(memory_dir, language))
        
    local_memory_qa = LocalMemoryRetrieval()

    local_memory_qa.init_cfg(
                            embedding_model="minilm-l6",
                            embedding_device="cuda",
                            top_k=args.memory_search_top_k,
                            language=language)

    # meta_prompt = generate_meta_prompt_dict_chatglm_app()[language]
    # new_user_meta_prompt = generate_new_user_meta_prompt_dict_chatglm()[language]
    
    only_related_prompt = only_related_prompt_dict_llava_app()[language]
    
    user_keyword = '[|User|]'
    ai_keyword = '[|LLaVA|]'
    
    # boot_name = boot_name_dict[language]
    boot_actual_name = "LLaVA"
    
    # memory_dir = '/13390024681/llama/EfficientVideo/Ours/memory_bank/memories.json'
    memory = json.loads(open(memory_dir,"r",encoding="utf-8").read())
    user_name = "User"
    # print(memory.keys())
    if user_name in memory.keys():
        if input('Would you like to summarize your memory? If yes, please enter "yes"') == "yes":
            print("Building ChtaGLM clinet !!!")
            config = {'max_tokens':1024, 'temperature':1, 'top_p':0.95, 'frequency_penalty':True}
            llm_client = LLMClientLLaMA3(gen_config=config, model_name='/13390024681/All_Model_Zoo/chatglm3-6b')
            print("ChtaGLM Client Building Finish !!!")
            user_memory = summarize_memory_event_personality(args, memory, user_name, llm_client=llm_client)
    hello_msg,user_memory,memory,user_name,user_memory_index = enter_name(user_name,memory,local_memory_qa,args)
    print(hello_msg)
    
    memory_config = {
        "user_memory":user_memory, 
        "user_name":user_name, 
        "user_memory_index":user_memory_index, 
        "local_memory_qa":local_memory_qa, 
        "only_related_prompt":only_related_prompt, 
        # "new_user_meta_prompt":new_user_meta_prompt, 
        "user_keyword":user_keyword, 
        "ai_keyword":ai_keyword, 
        "boot_actual_name":boot_actual_name, 
        "language":language,
        "memory": memory
    }
    
    print("Memory context retriveler building finished !!")
    
    output_list = []  # List to store the output results
    conv_mode = args.conv_mode

    video_formats = ['.mp4', '.avi', '.mov', '.mkv']

    # Iterate over each sample in the ground truth file
    index = 0
    sample_num = 100
    
    video_dir = args.video_dir
    if "msvd" in video_dir:
        data_mode = "MSVD"
    elif "MSRVTT" in args.video_dir:
        data_mode = "MSRVTT"
    elif "ActiveNet" in args.video_dir:
        data_mode = "ActiveNet"
    else:
        data_mode = "Others"
    
    if inference_mode == "on_line":
        video_path = args.video_dir
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_rate = int(cap.get(cv2.CAP_PROP_FPS))
        
        current_frame = 0
        num_frames = args.num_frames
        print("For present model, we only support {} frames video".format(num_frames))
        # User input to control the start of model inference
        
        while True:
            start_inference = input("Do you want to start inference? (yes/no): ").strip().lower()
            if start_inference == 'yes':
                break
            elif start_inference == 'no':
                print("Inference terminated by user.")
                return
        start_inference_triger = False
        finish_infernece = True
        
        # Create a queue to communicate with the user input thread
        input_queue = queue.Queue()
        pause_event = threading.Event()
        pause_event.set()  
        # Initially allow user input thread to run
        
        update_event = threading.Event()
        start_inference_event = threading.Event()
        update_event.set()
        # Ensure the memory update finished 
        
        # Start the user input thread
        # input_thread = threading.Thread(target=user_input_thread, args=(input_queue, pause_event))
                
        # Start the video reader thread
        video_thread = threading.Thread(target=video_reader_thread_with_embedding, args=(cap, total_frames, frame_rate, image_processor, model, current_frame))
        # video_thread = threading.Thread(target=video_reader_thread, args=(cap, frame_queue, total_frames, frame_rate))
        
        # Start the memory update thread
        update_thread = threading.Thread(target=updating_memory_buffer, args=(update_event, start_inference_event))
        # video_thread = threading.Thread(target=video_reader_thread, args=(cap, frame_queue, total_frames, frame_rate))
        
        # Start the inference thread
        infer_thread = threading.Thread(target=inference_thread_with_memory_and_dialogue_retrival_test, args=(start_inference_event, model, tokenizer, image_processor, args.num_frames, args.conv_mode, pause_event, update_event, chat, memory_config, args, current_frame, args.ppl))
        # infer_thread = threading.Thread(target=inference_thread_with_memory_test, args=(start_inference_event, model, tokenizer, image_processor, args.num_frames, args.conv_mode, pause_event, update_event, chat))
        # infer_thread = threading.Thread(target=inference_thread, args=(input_queue, frame_queue, model, tokenizer, image_processor, args.num_frames, args.conv_mode, pause_event, chat))
        
        video_thread.start()
        # input_thread.start()
        update_thread.start()
        infer_thread.start()

        video_thread.join()
        # input_thread.join()
        update_thread.join()
        infer_thread.join()


if __name__ == "__main__":
    args = parse_args()
    run_inference(args)