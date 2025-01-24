import math
import os
import argparse
import json
import torchaudio
import numpy as np

from tqdm import tqdm
from llava.eval.model_utils import load_video
from decord import VideoReader, cpu

from longva.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from longva.conversation import conv_templates, SeparatorStyle
from longva.model.builder import load_pretrained_model
# from llavanext.model.builder import load_pretrained_model  as load_llava_next 
from longva.utils import disable_torch_init
from longva.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path
from utiles import compute_gradients, Optical_flow, SSIM, long_short_memory_update, \
                    long_short_memory_update_with_summarize, search_tree, search_tree_multi_modal, \
                    fast_search_tree_multi_modal_with_embedding, MultimodalTreeNode, TreeNode, build_prompt_with_search_memory_only_related, \
                    convert_to_markdown, visualize_memory_feature_with_PCA, \
                    calculate_forgetting_probabilities, select_data_without_replacement, compress_spatial_features, weighted_kmeans_feature, \
                    fast_building_memory_tree_summarize_token, count_nodes_by_depth, RED, RESET, BLUE, GREEN# from llama_index.legacy.llms import (HuggingFaceLLM, CustomLLM, CompletionResponse, CompletionResponseGen, LLMMetadata)
# from llama_index.core.llms.callbacks import llm_completion_callback
from typing import Optional, List, Mapping, Any
from memory_bank.memory_retrieval.local_doc_qa import LocalMemoryRetrieval
from memory_bank.memory_utils import summarize_memory_event_personality, enter_name, save_local_memory
from memory_bank.summarize_memory import LLMClientLLaMA3
# from transformers import EvalPrediction, Trainer
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, AutoModel
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
import gc


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
condition = threading.Condition()
mutex = threading.Lock()
update_event = False
no_more_update = False
stop_all_thread_signal = False
long_memory_tree= None
buffer_cache = None
start_time = 0
end_time = 0

    
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
    parser.add_argument("--memory_file", type=str, required=False, default='updata_memories_for_streaming.json')
    parser.add_argument("--save_file", type=str, required=True, default='result_for_streaming.json')
    parser.add_argument("--annotations", type=str, required=True, default='result_for_streaming.json')
    parser.add_argument("--language", type=str, required=True, default='en')
    parser.add_argument("--memory_search_top_k", type=int, default=1)
    parser.add_argument("--ppl", action="store_true", help="weather to calculate ppl")
    parser.add_argument("--multi_modal_memory", action="store_true", help="weather to open multi-modal memory")
    

    return parser.parse_args()

def longva_inference_with_embedding(question, num_frames, conv_mode, model, tokenizer, chat, short_memory_buffer_cache, long_memory_tree_cache, history_prompt=None):
    
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

    # print("Question:{}".format(qs))
    
    conv = conv_templates[conv_mode].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()
    print("Prompt:{}".format(prompt))
    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt')
    input_ids = input_ids.unsqueeze(0).cuda()
    # ques_ids = ques_ids.unsqueeze(0).cuda()
    # print("input",input_ids)
    # print("ques",ques_ids)
    # print("image size 1 :{}".format(image_sizes))
    short_memory_embedding = torch.cat(short_memory_buffer_cache).view(-1, short_memory_buffer_cache[0].shape[-1]) # [4x576, 4096]
    print("short_memory_embedding", short_memory_embedding.shape)
    time_0 = time.time()
    # with threading.Lock():
    question_ids = tokenizer(question).input_ids
    question_embeddings  = model.get_model().embed_tokens(torch.tensor(question_ids, dtype=torch.long, device='cuda')) # num_text_token 4096
    if long_memory_tree_cache is not None:
        long_memory_list = search_tree(long_memory_tree_cache, torch.cat([question_embeddings, short_memory_embedding], dim=0))
        print("long memory list:{}".format(len(long_memory_list)))
        long_memory_embeddings = torch.cat(long_memory_list, dim=0).view(-1, long_memory_list[0].shape[-1]) # [40x36, 4096]
        print("long_memory_embeddings", long_memory_embeddings.shape)
        
        image_embeddings = torch.cat([short_memory_embedding, long_memory_embeddings], dim=0)
        # visualize_memory_feature_with_PCA(feature_bank, long_memory_list, clustering=5, same_color=False, only_most_important=False)
        
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
            modalities=["video"],
            # question_ids=ques_ids,
            # modalities="image",
            do_sample=True if args.temperature > 0 else False,
            temperature=args.temperature,
            top_p=args.top_p,
            num_beams=args.num_beams,
            max_new_tokens=512,
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

def longva_inference_with_embedding_multi_modal(question, num_frames, conv_mode, model, embedding_model, tokenizer, embedding_tokenizer, chat, short_memory_buffer_cache, long_memory_tree_cache, history_prompt=None):
    
    global feature_bank
    global start_time, end_time

    # ques_ids = ques_ids.unsqueeze(0).cuda()
    # print("input",input_ids)
    # print("ques",ques_ids)
    # print("image size 1 :{}".format(image_sizes))
    short_memory_embedding = torch.cat(short_memory_buffer_cache).view(-1, short_memory_buffer_cache[0].shape[-1]) # [4x576, 4096]
    time_0 = time.time()
    # with threading.Lock():
    question_ids = tokenizer(question).input_ids
    question_embeddings  = model.get_model().embed_tokens(torch.tensor(question_ids, dtype=torch.long, device='cuda')) # num_text_token 4096
    if long_memory_tree_cache is not None:
        long_memory_list, long_memory_text_list = fast_search_tree_multi_modal_with_embedding(long_memory_tree_cache, 
                                                                                              question, 
                                                                                              short_memory_embedding, 
                                                                                              embedding_model, 
                                                                                              embedding_tokenizer)
        # long_memory_list, long_memory_text_list = search_tree_multi_modal(long_memory_tree_cache, question_embeddings, short_memory_embedding, model, tokenizer)
        
        print("long memory list:{}".format(len(long_memory_list)))
        print("long memory text list:{}".format(len(long_memory_text_list)))
        
        long_memory_embeddings = torch.cat(long_memory_list, dim=0).view(-1, long_memory_list[0].shape[-1]) # [40x36, 4096]
        
        print("long_memory_embeddings", long_memory_embeddings.shape)
        print("short_memory_embedding", short_memory_embedding.shape)
        most_fine_grad_text = long_memory_text_list[-1]
        
        image_embeddings = torch.cat([short_memory_embedding, long_memory_embeddings], dim=0)
        # visualize_memory_feature_with_PCA(feature_bank, long_memory_list, clustering=5, same_color=False, only_most_important=False)
        
    else:
        image_embeddings = short_memory_embedding
        most_fine_grad_text = None
        # most_fine_grad_text = None
    
    # prm = "In addition, the text caption memory information articles most relevant to the current problem is '{most_fine_grad_text}'. \
    #     Please take advantage of the provided image embed and the previously mentioned contextual information to answer the following questions: "
    
    prm = "In addition, the text caption memory information articles most relevant to the current problem is '{most_fine_grad_text}'. \
        The image information you currently see and recall in the {image_token} is equally important as the contextual information mentioned earlier. \
        Sometimes the contextual information does not contain a direct answer to the question. \
        You need to synthesize this information and give an answer to the following question:"
        #  Please use the image information you currently see and recall at {image_token} as well as the previously mentioned contextual information to answer the following question: "
    notion = "DO NOT OUTPUT ANY EXPLANATORY TEXT THAT IS UNCERTAIN ABOUT THE CURRENT QUESTION."
    prm_wo_history = "For now you do not need history context to answer the following question:"
    
    if history_prompt is not None:
        if most_fine_grad_text is not None:
            if model.config.mm_use_im_start_end:
                qs = history_prompt + prm.format(most_fine_grad_text=most_fine_grad_text) + DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + question + notion
            else:
                qs = history_prompt + prm.format(most_fine_grad_text=most_fine_grad_text, image_token=DEFAULT_IMAGE_TOKEN) + '\n' + question + notion
        else:
            if model.config.mm_use_im_start_end:
                qs = history_prompt + DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + question + notion
            else:
                qs = history_prompt + '\n' + question + notion
    else:
        if model.config.mm_use_im_start_end:
            qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + question + notion
        else:
            qs = DEFAULT_IMAGE_TOKEN + '\n' + question + notion

    
    conv = conv_templates[conv_mode].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()
    print("Prompt:{}".format(prompt))
    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt')
    input_ids = input_ids.unsqueeze(0).cuda()
    
    
    time_1 = time.time()
    end_time = time.time()
    delay_time = end_time - start_time
    # print("delay time:{}".format(end_time - start_time))
    # all_delay.append((end_time - start_time))
    # print("image embedding shape:{}".format(image_embeddings.shape))
    # topk_images = [frame_list[idx] for idx in topk_indices]
    # for index, image in enumerate(topk_images):
    #     img = Image.fromarray(image)
    #     img.save("/13390024681/llama/EfficientVideo/Ours/save_images/topk_{}.jpg".format(topk_values[index]))
    
    with torch.inference_mode():
        output_ids = model.generate_with_image_embedding(
            input_ids,
            image_embeddings=[image_embeddings],
            modalities=["video"],
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
    
    print("process time:{}, generate time:{}".format(delay_time, (time_2 - time_1)))
    
    return outputs, delay_time, (time_2 - time_1)


def updating_memory_buffer(
    stop_all_thread_signal,
    start_inference_event, 
    video_reader_event,  
    summarizer_model, 
    summarizer_tokenzier, 
    building_multi_modal_memory_tree,
    short_window=20,
    remember_window=10,
    tau=5,
    compress_rate=1,
    chunk_size=30,
    num_clusters=5,
    interval=10):
    """
    Thread function to handle user input.
    """
    
    global short_memory_buffer
    global long_memory_tree
    global feature_bank
    global time_triger
    global buffer_cache
    global update_event
    global no_more_update
    
    captioning = "Please describe what you see in this video in as much detail as possible from a first-person perspective, including the surrounding environment, what objects are there, etc."
    # summarize = 
    
    if summarizer_model.config.mm_use_im_start_end:
        qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + captioning
    else:
        qs = DEFAULT_IMAGE_TOKEN + '\n' + captioning
            
    conv = conv_templates["qwen_1_5_ego"].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    captioning_prompt = conv.get_prompt()
    # print(question)
    captioning_input_ids = tokenizer_image_token(captioning_prompt, summarizer_tokenzier, IMAGE_TOKEN_INDEX, return_tensors='pt')
    captioning_input_ids = captioning_input_ids.unsqueeze(0).cuda()
    
    no_more_update = False
    # print("stop_all_thread_signal", stop_all_thread_signal)
    # while not stop_all_thread_signal:
    while not stop_all_thread_signal.is_set():
        # if not start_inference_triger and finish_infernece:
        
        # if len(feature_bank) > 20:
        # print("update_event", update_event)
        if update_event:
            print("<<<< start updating memory >>>>")
            # update_event.wait()
            # video_reader_event.clear()
            
            # buffer_cache
            # print("buffer_cache",len(buffer_cache))
            # assert 1==2
            # if len(buffer_cache) > 8:
            ############# building short memory ###################
            # print("<<<<<<< building short memory >>>>>>>>>>>")
            if len(buffer_cache) > short_window:
            # assert len(feature_bank) > short_window
                waite_FIFO = buffer_cache[-short_window:]
            else:
                short_window = len(buffer_cache)
                waite_FIFO = buffer_cache
            # assert len(waite_FIFO) > remember_window
            if remember_window > len(waite_FIFO):
                remember_window_set = len(waite_FIFO)
            else:
                remember_window_set =  remember_window
            
            forgetting_probs = calculate_forgetting_probabilities(short_window, tau=tau) # 注意需要调低tau的数值
            
            print("buffer_cache:{}".format(len(buffer_cache)))
            print("remember windows:{}".format(remember_window_set))
            print("waite_FIFO:{}".format(len(waite_FIFO)))
            print("forgetting_probs:{}".format(forgetting_probs))
            short_memory_buffer = select_data_without_replacement(waite_FIFO, forgetting_probs, remember_window_set)
            
            ############# building long memory with image captioning ###################
            
            # if compress_rate > 1:
            #     compressed_spatial_feature_list = compress_spatial_features(buffer_cache, compress_rate) # len 
            # else:
            #     compressed_spatial_feature_list = buffer_cache
            print("buffer_cache",len(buffer_cache))
            chunk_feature_list = [buffer_cache[i:i + chunk_size] for i in range(0, len(buffer_cache), chunk_size)] # length100 
            k_means_chunk_feature_list = [weighted_kmeans_feature(torch.cat(chunk_feature), num_clusters)[0] if len(chunk_feature)>= chunk_size else torch.cat(chunk_feature) for chunk_feature in chunk_feature_list] # length100 最后一个不需要聚类
            print("k_means_chunk_feature_list", k_means_chunk_feature_list[0].shape) #30 144 4096
            # print("chunk_feature_list", len(chunk_feature_list)) 
            # print("<<<<<<< building long memory tree >>>>>>>>>>>")
            long_memory_tree = fast_building_memory_tree_summarize_token(k_means_chunk_feature_list, 
                                                                            num_clusters, 
                                                                            interval, 
                                                                            summarizer_model, 
                                                                            captioning_input_ids, 
                                                                            summarizer_tokenzier, 
                                                                            chunk_feature_list,
                                                                            long_memory_tree)
            depth_count = count_nodes_by_depth(long_memory_tree)

            print("节点深度统计:")
            for depth, count in depth_count.items():
                print(f"{BLUE}深度 {depth}{RESET}: {count} 个节点")
            print("<<<<<<< memory update finish >>>>>>>>>>>")    
            # else:
            #     print("<<<< low cache mode not need long memory >>>>")
            #     short_memory_buffer = [buffer_cache[i] for i in range(len(buffer_cache))]
            #     long_memory_tree = None
            #     print("short_memory_buffer", len(short_memory_buffer))
            
            assert len(short_memory_buffer) > 0, "No memory ?"
            
            # update_event.clear()
            update_event = False
            # print("update_event", update_event)
            # start_inference_event.set()
            # video_reader_event.set()
        else:
            time.sleep(0.5)

    print("memory thread end here !!")
    no_more_update = True
    if long_memory_tree is not None:
        print(f"{GREEN}clear long memory tree{RESET}")
        long_memory_tree = None
    
    if buffer_cache is not None:
        print(f"{GREEN}clear buffer cache{RESET}")
        buffer_cache = None 
        
    if len(short_memory_buffer) > 0:
        print(f"{GREEN}clear short memory{RESET}")
        for cache in short_memory_buffer:
            del cache
    
    short_memory_buffer.clear()
    
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
 
def video_reader_thread_with_embedding(
    cap, 
    stop_all_thread_signal,
    total_frames, 
    frame_rate, 
    image_processor, 
    model, 
    count, 
    video_reader_event, 
    start_inference_event,
    time_line,
    chunk_size=30,
    ):
    """
    Thread function to read video frames and put them into a queue.
    """
    # 两个线程之间发生一定的资源互占的情况
    
    global frame_bank
    global feature_bank
    global time_count
    global time_triger
    global time_index
    global long_memory_tree
    global buffer_cache
    global update_event
    global start_time
    global short_memory_buffer
    
    current_frame_rate = 0
    # count = 0
    last_frame = None
    change_time = 0
    length = 0
    
    time_bank_1 = []
    time_bank_2 = []
    all_time_bank = []
    mag_bank = []
    
    pbar = tqdm(total=total_frames, desc="Processing Video", unit="frame")
    
    while cap.isOpened() and current_frame_rate < total_frames:
        # video_reader_event.wait()
        time_1 = time.time()
        ret, frame = cap.read()
        if not ret:
            break
        current_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # Save the last frame to image_show.jpg
        if time_count > 1:
            time_2 = time.time()
            # is_change, mean_mag, current_frame_tensor = SSIM(last_frame, current_frame, image_processor, model, 0.9) # judging by SSIM
            is_change, mean_mag, current_frame_tensor = Optical_flow(last_frame, current_frame, image_processor, model, 0.4) # judging by optical flow
            torch.cuda.empty_cache()
            time_3 = time.time()
            mag_bank.append(mean_mag)
            all_time_bank.append((time_3-time_2))
            if is_change:
                change_time += 1
                time_4 = time.time()
                image_embedding = model.encode_images(current_frame_tensor.unsqueeze(0).to(dtype=torch.float16))
                
                time_5 = time.time()
                # 确保有足够的帧数进行推理
                with mutex:
                    
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
                image_embedding = model.encode_images(current_frame_tensor.unsqueeze(0).to(dtype=torch.float16))
                # image_embedding = model.only_encode(current_frame_tensor.unsqueeze(0).to(dtype=torch.float16))
                # print("image embedding without proj:{}".format(image_embedding.shape)) # 1 576 1024
                # image_embedding = model.only_project(image_embedding)

                feature_bank.append(image_embedding)
            
        else:
            mean_mag = 0.0
            all_time_bank.append(0.000001)
       
       
        last_frame = current_frame
        # current_frame_rate += 1
        # with condition:
        # 先进行buffer数据的更新然后检测是否进行推理
        if len(feature_bank) >= chunk_size and len(feature_bank) % chunk_size == 0 and not update_event:
            
            # print("load vision buffer")
            # buffer_cache = buffer
            if long_memory_tree is not None:
                # print("last length:{}".format(length))
                # print("buffer length:{}".format(len(buffer)))
                buffer_cache = feature_bank[length:].copy() # out of index 
                # print("buffer_cahce length:{}".format(len(buffer_cahce)))
            else:
                buffer_cache = feature_bank.copy()
                # print("buffer_cahce length:{}".format(len(buffer_cahce)))
                length = len(buffer_cache)
            # print("vision buffer loaded ")
            # print(len(buffer_cache))
            # processed_length = length 
            # if length < len(feature_bank):
                # start_update = True
                
            if long_memory_tree is None: # 帮助判断如果该线程的速度快于主线程
                update_event = True
            elif long_memory_tree is not None and len(feature_bank) > length: # 检测当前进度是否需要更新
                update_event = True
                length += len(buffer_cache) 
            else:
                update_event = False
            
            
        time_count += 1
        if time_count in [t*frame_rate for t in time_line]:
            # update_event.set()
            if buffer_cache is None: # 如果对于刚开始的时候，feature bank 的长度没有达到要求，那就先进行直接copy一份
                short_memory_buffer  = feature_bank.copy()
            
            start_inference_event.set()
            time_index = time_index + 1
            start_time = time.time()
            
        time.sleep(0.015)
        time_7 = time.time()
        FPS = (time_count)/sum(all_time_bank)
        # print("FPS:{}".format(FPS))
        # Update tqdm progress bar and set postfix for FPS
        pbar.set_postfix(FPS="{:.2f}".format(FPS), MAG="{:.2f}".format(mean_mag), Time="{}".format(time_count), Buffer="{}".format(len(feature_bank)))
        pbar.update(1)
    # 通过构建完整的数据
    
    # del feature_bank
    # del mag_bank
    for tensor in feature_bank:
        del tensor
    feature_bank.clear()

    cap.release()
    stop_all_thread_signal.set()
    # print("stop_all_thread_signal", stop_all_thread_signal.is_set())
    print("Video processing completed.")
    print("Find chanement {} times in {}".format(change_time, total_frames))
    update_event = False
       
def inference_thread_with_memory_and_dialogue_retrival_test(
    start_inference_event, 
    stop_all_thread_signal,
    fps, 
    model, 
    embedding_model, 
    tokenizer, 
    embedding_tokenizer, 
    time_line, 
    num_frames, 
    conv_mode, 
    pause_event, 
    chat, 
    memory_config, 
    args, 
    current_frame, 
    save_file,
    question_list,
    output_loss=False):
    """
    Thread function to run inference on video frames.
    """
    global frame_bank
    global feature_bank
    global short_memory_buffer
    global long_memory_tree
    global time_triger
    global finish_triger
    global time_index    
    global update_event
    
    
    print("time_line", [t*fps for t in time_line])
    
            
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
    
    while not stop_all_thread_signal.is_set():

        if not update_event and start_inference_event.wait():
            index = time_index - 1
        # if start_inference_triger:
            with open(save_file, 'r', encoding='utf-8') as file:
                existing_data = json.load(file)
            # start_inference_triger = False
            # finish_infernece = False
            short_memory_buffer_cache = short_memory_buffer
            long_memory_tree_cache = long_memory_tree
            print("<<<< transfer finish and start to inference >>>>")
            question = question_list[index]['question']
            labels = question_list[index]['answer']
            qa_class = question_list[index]['class']
            count += 1
            if question.lower() == 'exit':
                print("Exit command received. Terminating inference.")
                break
            # print("<<<< transfer finish and start to inference >>>>")
            
            with torch.no_grad():
                print("<<<< Retrival Context >>>>")
                searched_history = build_prompt_with_search_memory_only_related(question, user_name, user_memory_index, local_memory_qa, only_related_prompt, user_keyword, ai_keyword, boot_actual_name)
                print("<<<< context retrieval finished  >>>>")
                # print("searched_history:{}".format(searched_history))

                
                output, process_time, generate_time = longva_inference_with_embedding_multi_modal(question, num_frames, 
                                                                                    conv_mode, model, embedding_model,
                                                                                    tokenizer, embedding_tokenizer, chat, 
                                                                                    short_memory_buffer_cache, long_memory_tree_cache, searched_history)
            print("LongVA:", output)
            existing_data.append({"time":time_line[index],"question":question,"label":labels,"predict":output,"class":qa_class, "process_time":process_time})
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
            
            with open(save_file, 'w', encoding='utf-8') as file:
                json.dump(existing_data, file, ensure_ascii=False, indent=4)

            time_triger = False
            pause_event.set()
            start_inference_event.clear()
            
            if len(time_line) == count:
                print("count fix, stop all thread")
                # stop_all_thread_signal.set()
                stop_all_thread_signal.set()
                print("stop_all_thread_signal", stop_all_thread_signal.is_set())
                break
            # update_event.set()
            # finish_infernece = True
            
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
            # if len(all_ppl) == len(question_list):
            #     print("Avg PPL:",sum(all_ppl)/len(all_ppl))
            #     differences = [abs(all_ppl[i+1] - all_ppl[i]) for i in range(len(all_ppl) - 1)]
            #     total_difference = sum(differences)
            #     count_differences = len(differences)
            #     average_difference = total_difference / count_differences
            #     print("Fluency PPL:", average_difference)
            
            # if stop_all_thread_signal.is_set():
            #     break
            # # else:
            # print("stop_all_thread_signal", stop_all_thread_signal.is_set())
            # print("update_event", update_event)
            time.sleep(1)
            
    print("inference thread end here !!")
                        
def run_inference(args):
    """
    Run inference on Video QA DataSetå.

    Args:
        args: Command-line arguments.
    """

    disable_torch_init()
    main_device = "cuda:0"
    
    inference_mode = args.mode
    model_path = os.path.expanduser(args.model_name)
    model_name = get_model_name_from_path(model_path)
    print("{}/{}".format(model_name, model_path))
    
    print("Initialize GPT-4o in LongVA-7B-DPO version:{} in {} mode !".format(args.conv_mode, inference_mode))
    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, None, "llava_qwen", device_map=main_device)
    
    print("Initialize GPT-4o in LongVA-7B-DPO version:{} in {} GPU1 !".format(args.conv_mode, inference_mode))
    _, model_cuda_1, _, _ = load_pretrained_model(model_path, None, "llava_qwen", device_map="cuda:1")
    print("model_cuda_1 device:{}".format(model_cuda_1.device))
    # 1. load model
    embedding_model_id = '/13390024681/All_Model_Zoo/mxbai-colbert-large-v1'
    embedding_tokenizer = AutoTokenizer.from_pretrained(embedding_model_id)
    embedding_model = AutoModel.from_pretrained(embedding_model_id).to(main_device)
    
    chat = None
    
    print("All model get ready !!")
    
    if args.ppl:
        print("Need to calculate PPL during test !!")
    
    all_annotations = json.load(open(args.annotations, 'r'))
    
    # data_path_dict = {""}
    video_dir = args.video_dir
    if "msvd" in video_dir:
        data_mode = "MSVD"
    elif "MSRVTT" in args.video_dir:
        data_mode = "MSRVTT"
    elif "ActiveNet" in args.video_dir:
        data_mode = "ActiveNet"
    else:
        data_mode = "Others"
    
    start = 0
    inference_count = 0
    infernece_limit = 4
    
    video_path_dict = {
        "/13390024681/All_Data/Streaming_final":["Cooking_show" ,"Comedy_drama" ,"Apple_TV"], 
        "/13390024681/All_Data/Supplement_1":["Cooking", "Metalworking"]
        }

    def find_key_by_category(category, data_dict):
        for key, categories in data_dict.items():
            if category in categories:
                return key
        return None
    
    # if inference_mode == "on_line":
    for anno in all_annotations:
        if inference_count < start:
            continue
        else:
            print("Building memory context retriveler !!")
            if not os.path.exists(args.memory_basic_dir):
                print("Building memory dir !!")
                os.mkdir(args.memory_basic_dir)
                
            args.memory_file = "memory_{}.json".format(inference_count)
            memory_dir = os.path.join(args.memory_basic_dir, args.memory_file)
            save_file = args.save_file
            # print("Save ",memory_dir)
            if not os.path.exists(memory_dir):
                json.dump({},open(memory_dir,"w",encoding="utf-8"))

            if not os.path.exists(save_file):
                json.dump([],open(save_file,"w",encoding="utf-8"))
                
            language = args.language
            print("Storing memory in {} using Language:{}".format(memory_dir, language))
                
            local_memory_qa = LocalMemoryRetrieval()

            local_memory_qa.init_cfg(
                                    embedding_model="minilm-l6",
                                    embedding_device=main_device,
                                    top_k=args.memory_search_top_k,
                                    language=language)
            
            only_related_prompt = only_related_prompt_dict_ego()[language]
            
            user_keyword = '[|User|]'
            ai_keyword = '[|AI|]'
            
            boot_actual_name = "AI"
            
            memory = json.loads(open(memory_dir,"r",encoding="utf-8").read())
            user_name = "User"
            
            hello_msg,user_memory,memory,user_name,user_memory_index = enter_name(user_name,memory,local_memory_qa,args)
            print(hello_msg)
            
            memory_config = {
                "user_memory":user_memory, 
                "user_name":user_name, 
                "user_memory_index":user_memory_index, 
                "local_memory_qa":local_memory_qa, 
                "only_related_prompt":only_related_prompt, 
                "user_keyword":user_keyword, 
                "ai_keyword":ai_keyword, 
                "boot_actual_name":boot_actual_name, 
                "language":language,
                "memory": memory
            }
            
            print("Memory context retriveler building finished !!")
        
            video_name = anno['info']['video_path']
            class_1 = anno['info']['class_1']
            class_2 = anno['info']['class_2']
            
            if class_1 == 'Ego':
                file_dir = "/13390024681/All_Data/EgoSchema/good_clips_git"
            else:
                file_dir = find_key_by_category(class_2, video_path_dict)
                file_dir = os.path.join(file_dir, class_2)
            
            question_list = anno['breakpoint']
            time_line = [int(ques['time']) for ques in question_list]
            
            video_path = os.path.join(file_dir, video_name)
            
            assert os.path.exists(video_path), "{} not exist ".format(video_path)
            # if not os.path.exists(video_path):
                # break        
            # video_path = args.video_dir
            cap = cv2.VideoCapture(video_path)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            frame_rate = int(cap.get(cv2.CAP_PROP_FPS))
            
            current_frame = 0
            
            num_frames = args.num_frames
            # print("For present model, we only support {} frames video".format(num_frames))
            # User input to control the start of model inference
            
            print("Start inference for all video !!")
            # start_inference_triger = False
            # finish_infernece = True
            
            # Create a queue to communicate with the user input thread
            pause_event = threading.Event()
            pause_event.set()  
            # Initially allow user input thread to run
            
            start_inference_event = threading.Event()
            start_inference_event.clear()
            
            video_reader_event = threading.Event()
            video_reader_event.set()
            
            stop_all_thread_signal = threading.Event()
            stop_all_thread_signal.clear()
            
            # Start the video reader thread
            video_thread = threading.Thread(target=video_reader_thread_with_embedding, 
                                            args=(
                                                cap, 
                                                stop_all_thread_signal,
                                                total_frames, 
                                                frame_rate, 
                                                image_processor, 
                                                model, 
                                                current_frame, 
                                                video_reader_event, 
                                                start_inference_event,
                                                time_line))
            video_thread.start()
            # video_thread = threading.Thread(target=video_reader_thread, args=(cap, frame_queue, total_frames, frame_rate))
            
            # Start the inference thread
            infer_thread = threading.Thread(target=inference_thread_with_memory_and_dialogue_retrival_test, 
                                            args=(
                                                start_inference_event, 
                                                stop_all_thread_signal,
                                                frame_rate, 
                                                model, 
                                                embedding_model,  
                                                tokenizer, 
                                                embedding_tokenizer, 
                                                time_line, 
                                                args.num_frames, 
                                                args.conv_mode, 
                                                pause_event, 
                                                chat, 
                                                memory_config, 
                                                args, 
                                                current_frame, 
                                                save_file, 
                                                question_list,
                                                args.ppl))
            # infer_thread = threading.Thread(target=inference_thread_with_memory_test, args=(start_inference_event, model, tokenizer, image_processor, args.num_frames, args.conv_mode, pause_event, update_event, chat))
            # infer_thread = threading.Thread(target=inference_thread, args=(input_queue, frame_queue, model, tokenizer, image_processor, args.num_frames, args.conv_mode, pause_event, chat))
            infer_thread.start()

            # Start the memory update thread
            update_thread = threading.Thread(target=updating_memory_buffer, 
                                            args=(
                                                stop_all_thread_signal,
                                                start_inference_event, 
                                                video_reader_event, 
                                                model_cuda_1, 
                                                tokenizer, 
                                                args.multi_modal_memory))
            update_thread.start()
            
            # stop_all_thread_signal.set()
            
            video_thread.join()
            # stop_all_thread_signal.set()
            infer_thread.join()
            update_thread.join()
            
            global time_count 
            global time_index
            # global long_memory_tree
            # global buffer_cache
            # global short_memory_buffer
            
            time_count = 0
            time_index = 0
            # 强制进行垃圾回收以确保资源被释放
            gc.collect()
            torch.cuda.empty_cache()
            
        inference_count += 1
        # if inference_count > infernece_limit:
        #     print("infernece finished !!")
            
        #     break


if __name__ == "__main__":
    args = parse_args()
    run_inference(args)