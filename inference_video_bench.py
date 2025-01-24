# import sys
# sys.path.append("/13390024681/llama/EfficientVideo/Ours")
import os
import argparse
import json
import numpy as np
import torch
import torch.nn.functional as F

from PIL import Image
import cv2
import time

from tqdm import tqdm
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
from utiles import process_images as process_images_ours
# from typing import Optional, List, Mapping, Any
# from memory_bank.memory_retrieval.local_doc_qa import LocalMemoryRetrieval
# from memory_bank.memory_utils import summarize_memory_event_personality, enter_name, save_local_memory
# from memory_bank.summarize_memory import LLMClientLLaMA3
# from transformers import EvalPrediction, Trainer
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, AutoModel
# from memory_bank.prompt_utils import *
    
def parse_args():
    """
    Parse command-line arguments.
    """
    parser = argparse.ArgumentParser()

    # Define the command-line arguments
    parser.add_argument('--video_dir', help='Directory containing video files.', required=False)
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
    # parser.add_argument("--memory_basic_dir", type=str, required=False, default='/13390024681/llama/EfficientVideo/Ours/memory_bank/memories')
    # parser.add_argument("--memory_file", type=str, required=False, default='updata_memories_for_streaming.json')
    # parser.add_argument("--save_file", type=str, required=True, default='result_for_streaming.json')
    # parser.add_argument("--annotations", type=str, required=True, default='result_for_streaming.json')
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
    
    # global feature_bank
    # global start_time, end_time

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
            
    conv = conv_templates["qwen_1_5_ego"].copy()
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

def get_seq_frames(total_num_frames, desired_num_frames):
    """
    Calculate the indices of frames to extract from a video.

    Parameters:
    total_num_frames (int): Total number of frames in the video.
    desired_num_frames (int): Desired number of frames to extract.

    Returns:
    list: List of indices of frames to extract.
    """

    # Calculate the size of each segment from which a frame will be extracted
    seg_size = float(total_num_frames - 1) / desired_num_frames

    seq = []
    for i in range(desired_num_frames):
        # Calculate the start and end indices of each segment
        start = int(np.round(seg_size * i))
        end = int(np.round(seg_size * (i + 1)))

        # Append the middle index of the segment to the list
        seq.append((start + end) // 2)

    return seq

def video_reader_thread_with_embedding(
    cap, 
    video_path,
    total_frames, 
    frame_rate, 
    image_processor, 
    model, 
    start,  # 新增，视频处理开始时间（秒）
    end,    # 新增，视频处理结束时间（秒）
    device,
    sample_rate,
    chunk_size=30,
    min_frame = 8,
    ):
    """
    Thread function to read video frames and put them into a queue.
    """
    if start is not None and end is not None:
        # Calculate start and end frames
        start_frame = int(start * frame_rate)
        end_frame = int(end * frame_rate)

        # Correct start and end frames to ensure they are within the video frame range
        start_frame = max(0, start_frame)
        end_frame = min(total_frames, end_frame)

        # Determine the number of frames to process
        total_frames_to_process = end_frame - start_frame
    else:
        start_frame = 0 
        end_frame = total_frames
        total_frames_to_process = end_frame - start_frame

    num_frame = int(total_frames_to_process * sample_rate)
    if num_frame < min_frame:
        num_frame = min_frame
        
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
    vr = VideoReader(video_path, ctx=cpu(0))
    total_frame_num = len(vr)

    # Currently, this function supports only 1 clip
    # assert n_clips == 1

    # Calculate total number of frames to extract
    total_num_frm = min(total_frame_num, num_frame)
    # Get indices of frames to extract
    frame_idx = get_seq_frames(total_frame_num, total_num_frm)
    # Extract frames as numpy array
    img_array = vr.get_batch(frame_idx).asnumpy()   # T H W C

    original_size = (img_array.shape[-2], img_array.shape[-3])  # (width, height)
    original_sizes = (original_size,) * total_num_frm

    clip_imgs = [Image.fromarray(img_array[j]) for j in range(total_num_frm)]
    
    # pbar = tqdm(total=len(frame_indices), desc="Processing Video", unit="frame")

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

    video_tensor = image_processor.preprocess(clip_imgs, return_tensors="pt")["pixel_values"].to(model.device, dtype=torch.float16)
    # print("video_tensor", video_tensor.shape)
    with torch.no_grad():
        # current_frame_tensor = image_processor(current_frame)
        # all_frame_tensor = torch.cat(frame_bank, dim=0).to(dtype=torch.float16)
        print("video_tensor:{}".format(video_tensor.shape))
        # current_frame_tensor= process_images_ours([Image.fromarray(current_frame)], image_processor, model.config).to(device).squeeze(0)
        image_embedding = model.encode_images(video_tensor)
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
    
    # print("time_line", [t*fps for t in time_line])
    
            
    process_time_bank = []
    generate_time_bank = []

    # if not update_event and start_inference_event.wait():
    # with open(save_file, 'r', encoding='utf-8') as file:
    #     existing_data = json.load(file)

    print("<<<< transfer finish and start to inference >>>>")
    
    with torch.no_grad():
        # searched_history = build_prompt_with_search_memory_only_related(question, user_name, user_memory_index, local_memory_qa, only_related_prompt, user_keyword, ai_keyword, boot_actual_name)
        print("<<<< context retrieval finished  >>>>")
        # print("searched_history:{}".format(searched_history))

        
        output, process_time, generate_time = longva_inference_with_embedding_multi_modal(
                                                question, num_frames, 
                                                conv_mode, model, embedding_model,
                                                tokenizer, embedding_tokenizer, chat, 
                                                short_memory_buffer_cache, long_memory_tree_cache, 
                                                None)
    print("Question:", question)
    print("LongVA:", output)
    # existing_data.append({"question":question,"label":labels,"predict":output,"class":qa_class, "process_time":process_time})
    process_time_bank.append(process_time)
    generate_time_bank.append(generate_time)
    
    torch.cuda.empty_cache()
    
    # with open(save_file, 'w', encoding='utf-8') as file:
    #     json.dump(existing_data, file, ensure_ascii=False, indent=4)

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
    
    print("Initialize LongVA-7B-DPO version:{} in {} mode !".format(args.conv_mode, inference_mode))
    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, None, "llava_qwen", device_map=main_device)
    
    print("Initialize LongVA-7B-DPO version:{} in {} GPU1 !".format(args.conv_mode, inference_mode))
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
    
    # data_path_dict = {""}
        
    # settings arguments for memory
    memory_chunk_size = args.chunk_size
    memory_num_clusters = args.num_clusters
    memory_interval = args.interval
    memory_short_window = args.short_window
    memory_remember_window = args.remember_window
    memory_tau = args.tau
    memory_compress_rate = args.compress_rate
    
    start = 0 # 191 219 239 217
    inference_count = 0
    infernece_limit = 4
    
    sample_rate = args.sample_rate
    print(f"{GREEN}Our sample rate :{RESET}", sample_rate)
    print(f"{GREEN}Start Inference from : {RESET}", start)
    # if inference_mode == "on_line":
    for dataset_name in tqdm(dataset_name_list, desc="StreamChat in Video_Bench"):
        qa_json = dataset_qajson[dataset_name]
        print(f'Dataset name:{dataset_name}, {qa_json=}!')
        with open(qa_json, 'r', encoding='utf-8') as f:
            all_annotations = json.load(f)
        
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
            # for anno in tqdm(all_annotations):
        
            if inference_count < start:
                inference_count += 1
                continue
            else:
                
                assert os.path.exists(video_path), "{} not exist ".format(video_path)
                # if not os.path.exists(video_path):
                    # break        
                # video_path = args.video_dir
                cap = cv2.VideoCapture(video_path)
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                frame_rate = int(cap.get(cv2.CAP_PROP_FPS))
                
                long_memory_tree = None
                
                print("Start inference for all video !!")
                # start_inference_triger = False
                # finish_infernece = True        
            
                feature_bank = video_reader_thread_with_embedding(
                    cap, 
                    video_path,
                    total_frames, 
                    frame_rate, 
                    image_processor, 
                    model, 
                    None, 
                    None,
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
                    None, 
                    args.num_frames, 
                    args.conv_mode, 
                    chat, 
                    None, 
                    args, 
                    None, 
                    question,
                    None,
                    None,
                    None,
                    args.ppl
                )
                
                # # update memory 
                # print("Update user memory !!")
                # b = [[question, output]]
                # # a, b = [[y[0], convert_to_markdown(y[1])] for y in history] ,history 
                # if user_name:
                #     memory = save_local_memory(memory,b,user_name,args)
                
                # _, user_memory,memory,user_name,user_memory_index = enter_name(user_name,memory,local_memory_qa,args)
                
                # memory_config["user_memory"] = user_memory
                # memory_config["memory"] = memory
                # memory_config["user_memory_index"] = user_memory_index
                
                eval_dict[q_id] = {
                    'video_id': video_id,
                    'question': question,
                    'output_sequence': output
                }  
                print(f'q_id:{q_id}, output:{output}!\n')
            
                torch.cuda.empty_cache()
                
                inference_count += 1
                # if inference_count > infernece_limit:
                #     print("infernece finished !!")
                    
                #     break
        # eval results
        eval_dataset_json = f'/13390024681/llama/EfficientVideo/Ours/output/Video_Bench/{dataset_name}_eval.json'
        with open(eval_dataset_json, 'w', encoding='utf-8') as f:
            json.dump(eval_dict, f, indent=2)

if __name__ == "__main__":
    args = parse_args()
    run_inference(args)