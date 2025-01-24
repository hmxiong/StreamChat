import math
import os
import argparse
import json
import torchaudio
import numpy as np
import random

from tqdm import tqdm
from llava.eval.model_utils import load_video
from decord import VideoReader, cpu

from vila.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from vila.conversation import conv_templates, SeparatorStyle
from vila.model.builder import load_pretrained_model
# from llavanext.model.builder import load_pretrained_model  as load_llava_next 
from vila.utils import disable_torch_init
from vila.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path
from utiles import compute_gradients, Optical_flow, SSIM, long_short_memory_update, \
                    long_short_memory_update_with_summarize, search_tree, search_tree_multi_modal, \
                    fast_search_tree_multi_modal_with_embedding, MultimodalTreeNode, TreeNode, build_prompt_with_search_memory_only_related, \
                    convert_to_markdown, visualize_memory_feature_with_PCA, \
                    calculate_forgetting_probabilities, select_data_without_replacement, compress_spatial_features, weighted_kmeans_feature, \
                    fast_building_memory_tree_summarize_token, count_nodes_by_depth, RED, RESET, BLUE, GREEN# from llama_index.legacy.llms import (HuggingFaceLLM, CustomLLM, CompletionResponse, CompletionResponseGen, LLMMetadata)
# from llama_index.core.llms.callbacks import llm_completion_callback
from utiles import process_images as process_images_ours
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
    # settongs for memory structure
    parser.add_argument("--chunk_size", type=int, default=20)
    parser.add_argument("--num_clusters", type=int, default=5)
    parser.add_argument("--interval", type=int, default=10)
    parser.add_argument("--short_window", type=int, default=20)
    parser.add_argument("--remember_window", type=int, default=5)
    parser.add_argument("--tau", type=int, default=5)
    parser.add_argument("--compress_rate", type=int, default=1)
    # settings for other part
    parser.add_argument("--num_chunks", type=int, default=1)
    parser.add_argument("--chunk_idx", type=int, default=0)
    parser.add_argument("--num_frames", type=int, default=4)
    parser.add_argument("--device", type=str, required=False, default='cuda:0')
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--sample_rate", type=float, default=0.5)
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

def longva_inference_with_embedding_multi_modal(question, num_frames, conv_mode, model, embedding_model, tokenizer, embedding_tokenizer, chat, short_memory_buffer_cache, long_memory_tree_cache, history_prompt=None):
    
    # global feature_bank
    # global start_time, end_time

    # ques_ids = ques_ids.unsqueeze(0).cuda()
    # print("input",input_ids)
    # print("ques",ques_ids)
    # print("image size 1 :{}".format(image_sizes))
    short_memory_embedding = torch.cat(short_memory_buffer_cache).view(-1, short_memory_buffer_cache[0].shape[-1]) # [4x576, 4096]
    time_0 = time.time()
    # with threading.Lock():
    # question_ids = tokenizer(question).input_ids
    # question_embeddings  = model.get_model().embed_tokens(torch.tensor(question_ids, dtype=torch.long, device='cuda')) # num_text_token 4096
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
    # delay_time = end_time - start_time
    delay_time = time_1 - time_0
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
            max_new_tokens=1024,
            use_cache=False)

    outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
    time_2 = time.time()
    # Perform inference and play the generated audio
    # wavs = chat.infer([outputs])
    # Audio(wavs[0], rate=24_000, autoplay=True)
    # Save the generated audio 
    # torchaudio.save("output.wav", torch.from_numpy(wavs[0]), 24000)
    
    time_3 = time.time() # TTS havy time dely
    
    print("process time:{}, generate time:{}".format(delay_time, (time_2 - time_1)))
    
    return outputs, delay_time, (time_2 - time_1)


def updating_memory_buffer(
    buffer_cache,
    long_memory_tree,
    summarizer_model, 
    summarizer_tokenzier, 
    building_multi_modal_memory_tree,
    short_window=20,
    remember_window=5,
    tau=5,
    compress_rate=1,
    chunk_size=30,
    num_clusters=5,
    interval=10):
    """
    Thread function to handle user input.
    """
    
    captioning = "Please describe what you see in this video in as much detail as possible from a first-person perspective, including the surrounding environment, what objects are there, etc."
    # summarize = 
    
    if summarizer_model.config.mm_use_im_start_end:
        qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + captioning
    else:
        qs = DEFAULT_IMAGE_TOKEN + '\n' + captioning
            
    conv = conv_templates["llama_3"].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    captioning_prompt = conv.get_prompt()
    # print(question)
    captioning_input_ids = tokenizer_image_token(captioning_prompt, summarizer_tokenzier, IMAGE_TOKEN_INDEX, return_tensors='pt')
    captioning_input_ids = captioning_input_ids.unsqueeze(0).cuda()
    
    # no_more_update = False
    # print("stop_all_thread_signal", stop_all_thread_signal)
    # while not stop_all_thread_signal:
    # while not stop_all_thread_signal.is_set():
    # if not start_inference_triger and finish_infernece:
    
    # if len(feature_bank) > 20:
    # print("update_event", update_event)
    # if update_event:
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
    # print("buffer_cache",len(buffer_cache))
    chunk_feature_list = [buffer_cache[i:i + chunk_size] for i in range(0, len(buffer_cache), chunk_size)] # length100 
    k_means_chunk_feature_list = [weighted_kmeans_feature(torch.cat(chunk_feature), num_clusters)[0] if len(chunk_feature)> chunk_size else torch.cat(chunk_feature) for chunk_feature in chunk_feature_list] # length100 最后一个不需要聚类
    # print("k_means_chunk_feature_list", k_means_chunk_feature_list[0].shape) #30 144 4096
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
    # update_event = False
    # print("update_event", update_event)
    # start_inference_event.set()
    # video_reader_event.set()
    return long_memory_tree, short_memory_buffer

# def video_reader_thread_with_embedding(
#     cap, 
#     total_frames, 
#     frame_rate, 
#     image_processor, 
#     model, 
#     start,  # 新增，视频处理开始时间（秒）
#     end,    # 新增，视频处理结束时间（秒）
#     device,
#     chunk_size=30,
#     ):
#     """
#     Thread function to read video frames and put them into a queue.
#     """
#     # 计算起始和结束帧
#     start_frame = int(start * frame_rate)
#     end_frame = int(end * frame_rate)

#     # 校正起始和结束帧，确保它们在视频帧范围内
#     start_frame = max(0, start_frame)
#     end_frame = min(total_frames, end_frame)

#     current_frame_number = 0
#     last_frame = None
#     change_time = 0
#     feature_bank = []

#     print(" Streat from {} and end with {}".format(start, end))
#     pbar = tqdm(total=(end_frame - start_frame), desc="Processing Video", unit="frame")

#     # 从指定的开始帧开始处理视频
#     cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

#     # for current_frame_number in range(start_frame, end_frame):
#     while current_frame_number < end_frame:
#         ret, frame = cap.read()
#         if not ret:
#             break

#         current_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#         if current_frame_number > start_frame:
#             # 使用光流法判断场景变化
#             with torch.no_grad():
#                 is_change, mean_mag, current_frame_tensor = Optical_flow(last_frame, current_frame, image_processor, model, 0.4)
#             torch.cuda.empty_cache()
            
#             if is_change:
#                 change_time += 1
#                 image_embedding = model.encode_images(current_frame_tensor.unsqueeze(0).to(dtype=torch.float16))
#                 feature_bank.append(image_embedding)
        
#         last_frame = current_frame
#         current_frame_number += 1
        
#         pbar.update(1)

#     pbar.close()
#     print("feature bank:{}".format(len(feature_bank)))
    
#     return feature_bank

def video_reader_thread_with_embedding(
    vr, 
    total_frames, 
    frame_rate, 
    image_processor, 
    model, 
    start,  # 新增，视频处理开始时间（秒）
    end,    # 新增，视频处理结束时间（秒）
    device,
    sample_rate,
    chunk_size=30,
    ):
    """
    Thread function to read video frames and put them into a queue.
    """
    # Calculate start and end frames
    start_frame = int(start * frame_rate)
    end_frame = int(end * frame_rate)

    # Correct start and end frames to ensure they are within the video frame range
    start_frame = max(0, start_frame)
    end_frame = min(total_frames, end_frame)

    # Determine the number of frames to process
    total_frames_to_process = end_frame - start_frame

    num_frame = int(total_frames_to_process * sample_rate)
    print(f"{GREEN}num frame{RESET}", num_frame)
    print(f"{GREEN}total_frames_to_process{RESET}", total_frames_to_process)
    
    if num_frame > 900:
        num_frame = 200 # need constrict
        
    # Decide whether to sample or use every frame
    if total_frames_to_process <= chunk_size:
        # Use every frame between start and end
        frame_indices = range(start_frame, end_frame)
    else:
        # Calculate frame indices for equal interval sampling
        frame_indices = [
            int(start_frame + i * total_frames_to_process / num_frame)
            for i in range(num_frame)
        ]

    # feature_bank = []
    frame_bank = []

    print("Starting from {} and ending with {}".format(start, end))
    # pbar = tqdm(total=len(frame_indices), desc="Processing Video", unit="frame")
    
    # vr
    img_array = vr.get_batch(frame_indices).asnumpy()
    # clip_imgs = [Image.fromarray(img_array[j]) for j in range(len(frame_indices))]
    print("Start processing videos !!")
    for image in img_array:
        # image = expand2square(image, tuple(int(x * 255) for x in image_processor.image_mean))
        image = image_processor.preprocess(image, return_tensors="pt")["pixel_values"][0].to(device)
        # print(image.shape)
        frame_bank.append(image.unsqueeze(0)) # [ dim resize_w resize_h ]
        # pbar.update(1)
        
    # for current_frame_number in frame_indices:
    #     # Set the position to the current frame number
    #     cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame_number)
    #     ret, frame = cap.read()
    #     if not ret:
    #         break

    #     current_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    #     current_frame_tensor= process_images_ours([Image.fromarray(current_frame)], image_processor, model.config).to(device)
        
    #     frame_bank.append(current_frame_tensor)
    #     # Directly encode the current frame
        
    #     pbar.update(1)

    with torch.no_grad():
        # current_frame_tensor = image_processor(current_frame)
        all_frame_tensor = torch.cat(frame_bank, dim=0).to(dtype=torch.float16)
        print("\n all_frame_tensor:{}".format(all_frame_tensor.shape))
        # current_frame_tensor= process_images_ours([Image.fromarray(current_frame)], image_processor, model.config).to(device).squeeze(0)
        image_embedding = model.encode_images(all_frame_tensor)
        bs = image_embedding.shape[0]
        feature_bank = [image_embedding[i:i+1] for i in range(bs)]
    
    
    # pbar.close()
    print("Feature bank size: {}".format(len(feature_bank)))
    assert len(feature_bank) == bs  
    return feature_bank
    
# def video_reader_thread_with_embedding_sample(
#     cap, 
#     total_frames, 
#     frame_rate, 
#     image_processor, 
#     model, 
#     start,  # Video processing start time (in seconds)
#     end,    # Video processing end time (in seconds)
#     num_frame,  # Number of frames to sample
#     device
# ):
#     """
#     Thread function to read video frames and put them into a queue.
#     """
#     # Calculate start and end frames
#     start_frame = int(start * frame_rate)
#     end_frame = int(end * frame_rate)

#     # Correct start and end frames to ensure they are within the video frame range
#     start_frame = max(0, start_frame)
#     end_frame = min(total_frames, end_frame)

#     # Calculate frame indices for equal interval sampling
#     frame_indices = [
#         int(start_frame + i * (end_frame - start_frame) / num_frame)
#         for i in range(num_frame)
#     ]

#     feature_bank = []

#     print("Starting from {} and ending with {}".format(start, end))
#     pbar = tqdm(total=len(frame_indices), desc="Processing Video", unit="frame")

#     for current_frame_number in frame_indices:
#         # Set the position to the current frame number
#         cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame_number)
#         ret, frame = cap.read()
#         if not ret:
#             break

#         current_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

#         # Directly encode the current frame
#         with torch.no_grad():
#             current_frame_tensor = image_processor(current_frame)
#             image_embedding = model.encode_images(current_frame_tensor.unsqueeze(0).to(device=device, dtype=torch.float16))
#             feature_bank.append(image_embedding)
        
#         pbar.update(1)

#     pbar.close()
#     print("Feature bank size: {}".format(len(feature_bank)))
    
#     return feature_bank
       
def inference_thread_with_memory_and_dialogue_retrival_test(
    long_memory_tree_cache,
    short_memory_buffer_cache,
    fps, 
    model, 
    embedding_model, 
    tokenizer, 
    embedding_tokenizer, 
    time_line, 
    num_frames, 
    conv_mode, 
    chat, 
    memory_config, 
    args, 
    save_file,
    question,
    labels,
    qa_class,
    time,
    output_loss=False):
    """
    Thread function to run inference on video frames.
    """
    
    print("time_line", [t*fps for t in time_line])
    
            
    process_time_bank = []
    generate_time_bank = []
    # time_step = []
    # all_ppl = []
    # avg_ppl = []
    # count = 0
        
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
    
    # while not stop_all_thread_signal.is_set():

    # if not update_event and start_inference_event.wait():
    # if start_inference_triger:
    with open(save_file, 'r', encoding='utf-8') as file:
        existing_data = json.load(file)
    # start_inference_triger = False
    # finish_infernece = False
    # short_memory_buffer_cache = short_memory_buffer
    # long_memory_tree_cache = long_memory_tree
    print("<<<< transfer finish and start to inference >>>>")
    
    with torch.no_grad():
        print("<<<< Retrival Context >>>>")
        searched_history = build_prompt_with_search_memory_only_related(question, user_name, user_memory_index, local_memory_qa, only_related_prompt, user_keyword, ai_keyword, boot_actual_name)
        print("<<<< context retrieval finished  >>>>")
        # print("searched_history:{}".format(searched_history))

        
        output, process_time, generate_time = longva_inference_with_embedding_multi_modal(question, num_frames, 
                                                                            conv_mode, model, embedding_model,
                                                                            tokenizer, embedding_tokenizer, chat, 
                                                                            short_memory_buffer_cache, long_memory_tree_cache, searched_history)
    print("VILA:", output)
    existing_data.append({"time":time,"question":question,"label":labels,"predict":output,"class":qa_class, "process_time":process_time})
    process_time_bank.append(process_time)
    generate_time_bank.append(generate_time)
    # time_step.append(length)
    
    # print("LLaVA:", output)
    
    torch.cuda.empty_cache()
    
    with open(save_file, 'w', encoding='utf-8') as file:
        json.dump(existing_data, file, ensure_ascii=False, indent=4)

    return output
    
                        
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
    
    print("Initialize GPT-4o in VILA1.5 version:{} in {} mode !".format(args.conv_mode, inference_mode))
    # tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, None, "llava_qwen", device_map=main_device)
    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, model_name, None, device_map=main_device)
    print("model device:{}".format(model.device))
    
    print("Initialize GPT-4o in VILA1.5 version:{} in {} GPU1 !".format(args.conv_mode, inference_mode))
    # _, model_cuda_1, _, _ = load_pretrained_model(model_path, None, "llava_qwen", device_map="cuda:1")
    _, model_cuda_1, _, _  = load_pretrained_model(model_path, model_name, None, device_map="cuda:1")
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
    # random.shuffle(all_annotations)
    
    # data_path_dict = {""}
    video_dir = args.video_dir
    
    if "msvd" in video_dir:
        data_mode = "MSVD"
    elif "MSRVTT" in args.video_dir:
        data_mode = "MSRVTT"
    elif "ActiveNet" in args.video_dir:
        data_mode = "ActiveNet"
    elif "Streaming" in args.video_dir:
        data_mode = "Streaming"
    else:
        data_mode = "Others"
        
    # settings arguments for memory
    memory_chunk_size = args.chunk_size
    memory_num_clusters = args.num_clusters
    memory_interval = args.interval
    memory_short_window = args.short_window
    memory_remember_window = args.remember_window
    memory_tau = args.tau
    memory_compress_rate = args.compress_rate
    
    start = 100 # 191 219 239 217
    inference_count = 0
    infernece_limit = 4
    
    sample_rate = args.sample_rate
    print(f"{GREEN}Our sample rate :{RESET}", sample_rate)
    print(f"{GREEN}Start Inference from : {RESET}", start)
    # if inference_mode == "on_line":
    for anno in tqdm(all_annotations):
        if inference_count < start:
            inference_count += 1
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

            file_dir = os.path.join(video_dir, class_1)
            
            question_list = anno['breakpoint']
            time_line = [int(ques['time']) for ques in question_list]
            
            video_path = os.path.join(file_dir, video_name)
            
            assert os.path.exists(video_path), "{} not exist ".format(video_path)
            # if not os.path.exists(video_path):
                # break        
            # video_path = args.video_dir
            # cap = cv2.VideoCapture(video_path)
            # total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            # frame_rate = int(cap.get(cv2.CAP_PROP_FPS))
            
            vr = VideoReader(video_path, ctx=cpu(0))
            total_frame_num = len(vr)
            frame_rate = vr.get_avg_fps()
            
            # current_frame = 0
            frame_line = [0] + time_line
            
            long_memory_tree = None

            print("time line :{}".format(time_line))
            
            for questions, star, end in zip(question_list, frame_line[:-1], frame_line[1:]):
                
                question = questions["question"]
                labels = questions["answer"]
                qa_class= questions["class"]
                num_frames = args.num_frames
                # print("For present model, we only support {} frames video".format(num_frames))
                # User input to control the start of model inference
                
                print("Start inference for all video !!")
                # start_inference_triger = False
                # finish_infernece = True        
                
                feature_bank = video_reader_thread_with_embedding(
                    vr, 
                    total_frame_num, 
                    frame_rate, 
                    image_processor, 
                    model, 
                    star, 
                    end,
                    main_device,
                    sample_rate,
                    chunk_size=memory_chunk_size
                )
                
                if len(feature_bank) > 0:
                    long_memory_tree, short_memory_buffer = updating_memory_buffer(
                        feature_bank,
                        long_memory_tree,
                        model_cuda_1, 
                        tokenizer, 
                        args.multi_modal_memory,
                        short_window=memory_short_window,
                        remember_window=memory_remember_window,
                        tau=memory_tau,
                        compress_rate=memory_compress_rate,
                        chunk_size=memory_chunk_size,
                        num_clusters=memory_num_clusters,
                        interval=memory_interval
                    )
                
                output = inference_thread_with_memory_and_dialogue_retrival_test(
                    long_memory_tree,
                    short_memory_buffer,
                    frame_rate, 
                    model, 
                    embedding_model,  
                    tokenizer, 
                    embedding_tokenizer, 
                    time_line, 
                    args.num_frames, 
                    args.conv_mode, 
                    chat, 
                    memory_config, 
                    args, 
                    save_file, 
                    question,
                    labels,
                    qa_class,
                    end,
                    args.ppl
                )
                
                # update memory 
                print("Update user memory !!")
                b = [[question, output]]
                # a, b = [[y[0], convert_to_markdown(y[1])] for y in history] ,history 
                if user_name:
                    memory = save_local_memory(memory,b,user_name,args)
                
                _, user_memory,memory,user_name,user_memory_index = enter_name(user_name,memory,local_memory_qa,args)
                
                memory_config["user_memory"] = user_memory
                memory_config["memory"] = memory
                memory_config["user_memory_index"] = user_memory_index
                
               
                # torch.cuda.empty_cache()
                
            inference_count += 1
            # if inference_count > infernece_limit:
            #     print("infernece finished !!")
                
            #     break


if __name__ == "__main__":
    args = parse_args()
    run_inference(args)