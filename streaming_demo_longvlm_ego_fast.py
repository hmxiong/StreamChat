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
                    fast_building_memory_tree_summarize_token, count_nodes_by_depth, RED, RESET, BLUE
# from llama_index.legacy.llms import (HuggingFaceLLM, CustomLLM, CompletionResponse, CompletionResponseGen, LLMMetadata)
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


# Global variables for shared resources
# frame_bank = queue.Queue()
# feature_bank = queue.Queue()
frame_bank   = []
feature_bank = []
short_memory_buffer = []
buffer_cache = None
time_count = 0
time_index = 0
time_triger = False
finish_triger = True
update_event = False
stop_all_thread_signal = False
long_memory_tree= None
condition = threading.Condition()
mutex = threading.Lock()
start_time = 0
end_time = 0
all_delay = []

question_list = [
            {
                "question": "Is the person in the video wearing glasses?",
                "answer": "Yes, the person is wearing glasses.",
                "class": "ST",
                "time": 3
            },
            {
                "question": "Where is the green sofa?",
                "answer": "The green sofa is positioned against the wall near the black sofa.",
                "class": "FT",
                "time": 45
            },
            {
                "question": "From which animated series does the character Garfield originate?",
                "answer": "The character Garfield originates from the animated series 'Garfield'. ",
                "class": "KG",
                "time": 76
            },
            {
                "question": "Based on what you see, where is the blue bookshelf?",
                "answer": "The blue bookshelf is positioned next to the curtains, against the wall.",
                "class": "LM",
                "time": 150
            },
            {
                "question": "Where is the guitar?",
                "answer": "The guitar is on the bed.",
                "class": "SM",
                "time": 226
            },
            {
                "question": "Are there any musical instruments in this video?",
                "answer": "Yes, there is a guitar on the bed.",
                "class": "CS",
                "time": 280
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
    parser.add_argument("--save_file", type=str, required=True, default='result_for_streaming.json')
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

def longva_inference_with_embedding_multi_modal(
    question, 
    num_frames, 
    conv_mode, 
    model, 
    embedding_model, 
    tokenizer, 
    embedding_tokenizer, 
    chat, 
    short_memory_buffer_cache, 
    long_memory_tree_cache, 
    history_prompt=None):
    
    global feature_bank
    global start_time, end_time, all_delay

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
        if model.config.mm_use_im_start_end:
            qs = history_prompt + prm.format(most_fine_grad_text=most_fine_grad_text) + DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + question + notion
        else:
            qs = history_prompt + prm.format(most_fine_grad_text=most_fine_grad_text, image_token=DEFAULT_IMAGE_TOKEN) + '\n' + question + notion
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
    print("delay time:{}".format(end_time - start_time))
    all_delay.append((end_time - start_time))
    
    if len(all_delay) == len(question_list):
        with open('./delay.json', 'w') as f:
            json.dump(all_delay, f, indent=4)
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
    
    print("process time:{}, generate time:{}".format((time_1 - time_0), (time_2 - time_1)))
    
    return outputs, (time_1 - time_0), (time_2 - time_1)

def longva_inference_with_embedding_and_ppl(question, label, num_frames, conv_mode, model, tokenizer, chat, short_memory_buffer_cache, long_memory_tree_cache, history_prompt=None):
    
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
    short_memory_embedding = torch.cat(short_memory_buffer_cache).view(-1, short_memory_buffer_cache[0].shape[-1]) # [4x144, 4096]
    
    time_0 = time.time()
    # with threading.Lock():
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


def updating_memory_buffer(
    start_inference_event, 
    video_reader_event,  
    summarizer_model, 
    summarizer_tokenzier, 
    building_multi_modal_memory_tree,
    short_window=20,
    remember_window=10,
    tau=10,
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
    global stop_all_thread_signal
    global buffer_cache
    global update_event
    
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
    
    
    # print("stop_all_thread_signal", stop_all_thread_signal)
    while not stop_all_thread_signal:
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
                remember_window = len(waite_FIFO)
                
            forgetting_probs = calculate_forgetting_probabilities(short_window, tau=tau)
            short_memory_buffer = select_data_without_replacement(waite_FIFO, forgetting_probs, remember_window)
            
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
    
def user_input_thread(
    input_queue, 
    pause_event,
    frame_rate):
    """
    Thread function to handle user input.
    """
    global time_index
    global time_count
    
    while True:
        if time_count in [t*frame_rate for t in time_line]:
            # update_event.set()
            time_index = time_index + 1
 
def video_reader_thread_with_embedding(
    cap, 
    total_frames, 
    frame_rate, 
    image_processor, 
    model, 
    count, 
    video_reader_event, 
    start_inference_event,
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
    global stop_all_thread_signal
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
            is_change, mean_mag, current_frame_tensor = Optical_flow(last_frame, current_frame, image_processor, model, 0.25) # judging by optical flow
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
        
        if len(feature_bank) >= chunk_size and len(feature_bank) % chunk_size == 0 and not update_event:
            # print("load vision buffer")
            # buffer_cache = buffer
            if long_memory_tree is not None:
                # print("last length:{}".format(length))
                # print("buffer length:{}".format(len(buffer)))
                buffer_cache = feature_bank[length:].copy() # out of index 
                # print("buffer_cahce length:{}".format(len(buffer_cahce)))
                length += len(buffer_cache) 
                
            else:
                buffer_cache = feature_bank.copy()
                # print("buffer_cahce length:{}".format(len(buffer_cahce)))
                length = len(buffer_cache)
            # print("vision buffer loaded ")
            # print(len(buffer_cache))
            # processed_length = length 
            # if length < len(feature_bank):
                # start_update = True
            update_event = True
            # print("update_event", update_event)
            # else:
            #     start_update = False
            
        time_count += 1
        if time_count in [t*frame_rate for t in time_line]:
            # update_event.set()
            if buffer_cache is None: # 如果没有达到更新条件，就先临时更新一下short_memory_buffer
                short_memory_buffer = feature_bank.copy()
                
            start_inference_event.set()
            time_index = time_index + 1
            start_time = time.time()
            
        time.sleep(0.01)
        time_7 = time.time()
        FPS = (time_count)/sum(all_time_bank)
        # print("FPS:{}".format(FPS))
        # Update tqdm progress bar and set postfix for FPS
        pbar.set_postfix(FPS="{:.2f}".format(FPS), MAG="{:.2f}".format(mean_mag), Time="{}".format(time_count), Buffer="{}".format(len(feature_bank)))
        pbar.update(1)
    
    stop_all_thread_signal = True    
    cap.release()
    print("Video processing completed.")
    print("Find chanement {} times in {}".format(change_time, total_frames))
       
def inference_thread_with_memory_and_dialogue_retrival_test(
    start_inference_event, 
    fps, 
    model, 
    embedding_model, 
    tokenizer, 
    embedding_tokenizer, 
    image_processor, 
    num_frames, 
    conv_mode, 
    pause_event, 
    chat, 
    memory_config, 
    args, 
    current_frame, 
    save_file, 
    output_loss=False):
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
    global stop_all_thread_signal
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
    
    while not stop_all_thread_signal:
        # print()
        # if not input_queue.empty():
        # if frame_bank.qsize() % 15 == 0 and frame_bank.qsize() != 0:
        # if len(feature_bank) % 40 == 0 and len(feature_bank) != 0:
        # with condition:
        #     condition.wait()  # 等待x值的更新通知
        #     if x in list_data:
        # if time_count in 
        if not update_event and start_inference_event.wait():
            index = time_index - 1
            # update_event.clear()
        # if start_inference_triger:
            # start_inference_event.wait()
            with open(save_file, 'r', encoding='utf-8') as file:
                existing_data = json.load(file)
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
            
            print("<<<< Retrival Context >>>>")
            searched_history = build_prompt_with_search_memory_only_related(question, user_name, user_memory_index, local_memory_qa, only_related_prompt, user_keyword, ai_keyword, boot_actual_name)
            print("<<<< context retrieval finished  >>>>")
            # print("searched_history:{}".format(searched_history))

            # else:
            output, process_time, generate_time = longva_inference_with_embedding_multi_modal(question, num_frames, 
                                                                                conv_mode, model, embedding_model,
                                                                                tokenizer, embedding_tokenizer, chat, 
                                                                                short_memory_buffer_cache, long_memory_tree_cache, searched_history)
            print("LongVA:", output)
            existing_data.append({"time":index,"question":question,"label":labels,"predict":output})
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
            time_triger = False
            # pause_event.set()
            start_inference_event.clear()
            with open(save_file, 'w', encoding='utf-8') as file:
                json.dump(existing_data, file, ensure_ascii=False, indent=4)

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
            if len(all_ppl) == len(question_list):
                print("Avg PPL:",sum(all_ppl)/len(all_ppl))
                differences = [abs(all_ppl[i+1] - all_ppl[i]) for i in range(len(all_ppl) - 1)]
                total_difference = sum(differences)
                count_differences = len(differences)
                average_difference = total_difference / count_differences
                print("Fluency PPL:", average_difference)
                
            time.sleep(0.01)
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
    print("model_cuda_0 device:{}".format(model.device))
    
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
        
    print("Building memory context retriveler !!")
    memory_dir = os.path.join(args.memory_basic_dir,args.memory_file)
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

    # meta_prompt = generate_meta_prompt_dict_chatglm_app()[language]
    # new_user_meta_prompt = generate_new_user_meta_prompt_dict_chatglm()[language]
    
    only_related_prompt = only_related_prompt_dict_ego()[language]
    
    user_keyword = '[|User|]'
    ai_keyword = '[|AI|]'
    
    # boot_name = boot_name_dict[language]
    boot_actual_name = "AI"
    
    # memory_dir = '/13390024681/llama/EfficientVideo/Ours/memory_bank/memories.json'
    memory = json.loads(open(memory_dir,"r",encoding="utf-8").read())
    user_name = "User"
    # print(memory.keys())
    # if user_name in memory.keys():
    #     if input('Would you like to summarize your memory? If yes, please enter "yes"') == "yes":
    #         print("Building ChtaGLM clinet !!!")
    #         config = {'max_tokens':1024, 'temperature':1, 'top_p':0.95, 'frequency_penalty':True}
    #         llm_client = LLMClientLLaMA3(gen_config=config, model_name='/13390024681/All_Model_Zoo/chatglm3-6b')
    #         print("ChtaGLM Client Building Finish !!!")
    #         user_memory = summarize_memory_event_personality(args, memory, user_name, llm_client=llm_client)
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
        
        # while True:
        #     start_inference = input("Do you want to start inference? (yes/no): ").strip().lower()
        #     if start_inference == 'yes':
        #         break
        #     elif start_inference == 'no':
        #         print("Inference terminated by user.")
        #         return
        # start_inference_triger = False
        # finish_infernece = True
        
        # Create a queue to communicate with the user input thread
        input_queue = queue.Queue()
        pause_event = threading.Event()
        pause_event.set()  
        # Initially allow user input thread to run
        
        # update_event = threading.Event()
        # update_event = False
        start_inference_event = threading.Event()
        video_reader_event = threading.Event()
        # update_event.clear()
        video_reader_event.set()
        start_inference_event.clear()
        # Ensure the memory update finished 
        
        # Start the user input thread
        # input_thread = threading.Thread(target=user_input_thread, args=(input_queue, pause_event))
                
        # Start the video reader thread
        video_thread = threading.Thread(target=video_reader_thread_with_embedding, args=(cap, 
                                                                                         total_frames, 
                                                                                         frame_rate, 
                                                                                         image_processor, 
                                                                                         model, 
                                                                                         current_frame, 
                                                                                         video_reader_event,
                                                                                         start_inference_event 
                                                                                         ))
        # video_thread = threading.Thread(target=video_reader_thread, args=(cap, frame_queue, total_frames, frame_rate))
        
        # Start the memory update thread
        update_thread = threading.Thread(target=updating_memory_buffer, args=(start_inference_event, 
                                                                              video_reader_event, 
                                                                              model_cuda_1, 
                                                                              tokenizer, 
                                                                              args.multi_modal_memory
                                                                              ))
        # video_thread = threading.Thread(target=video_reader_thread, args=(cap, frame_queue, total_frames, frame_rate))
        
        # Start the inference thread
        infer_thread = threading.Thread(target=inference_thread_with_memory_and_dialogue_retrival_test, args=(start_inference_event, 
                                                                                                              frame_rate, 
                                                                                                              model, 
                                                                                                              embedding_model,  
                                                                                                              tokenizer, 
                                                                                                              embedding_tokenizer, 
                                                                                                              image_processor, 
                                                                                                              args.num_frames, 
                                                                                                              args.conv_mode, 
                                                                                                              pause_event, 
                                                                                                              chat, 
                                                                                                              memory_config, 
                                                                                                              args, 
                                                                                                              current_frame, 
                                                                                                              save_file, 
                                                                                                              args.ppl))
        # infer_thread = threading.Thread(target=inference_thread_with_memory_test, args=(start_inference_event, model, tokenizer, image_processor, args.num_frames, args.conv_mode, pause_event, update_event, chat))
        # infer_thread = threading.Thread(target=inference_thread, args=(input_queue, frame_queue, model, tokenizer, image_processor, args.num_frames, args.conv_mode, pause_event, chat))
        
        video_thread.start()
        update_thread.start()
        infer_thread.start()

        video_thread.join()
        update_thread.join()
        infer_thread.join()


if __name__ == "__main__":
    args = parse_args()
    run_inference(args)