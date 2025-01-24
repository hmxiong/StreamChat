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
from utiles import compute_gradients, Optical_flow, SSIM, long_short_memory_update, long_short_memory_update_with_summarize, search_tree, search_tree_multi_modal, search_tree_multi_modal_with_embedding, MultimodalTreeNode, TreeNode, build_prompt_with_search_memory_only_related, convert_to_markdown, visualize_memory_feature_with_PCA
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
long_memory_tree= MultimodalTreeNode
condition = threading.Condition()
mutex = threading.Lock()

# question_list = [
#             {
#                 "question": "Can you tell me if the current scene in the video takes place at the sea or on land?",
#                 "answer": "Sure, The scene in the video takes place at the sea.",
#                 "class": "ST",
#                 "time": 28
#             },
#             {
#                 "question": "I need a tool to ladle soup, can you help me find out the appropriate tool in the current video?",
#                 "answer": "Yes, The ladle in the hand of the bald man in the video is a good choice.",
#                 "class": "FT",
#                 "time": 41
#             },
#             {
#                 "question": "Can you tell me where the bottle containing the golden-yellow liquid, which the man held in his left hand, was placed previously?",
#                 "answer": "The bottle containing the golden-yellow liquid is on the table in front of the man, next to a brown or black bottle.",
#                 "class": "SM",
#                 "time": 109
#             },
#             {
#                 "question": "Where is the porcelain capital?",
#                 "answer": "The porcelain capital is Jingdezhen, located in Jiangxi Province, China.",
#                 "class": "KG",
#                 "time": 143
#             },
#             {
#                 "question": "Can you tell me where the plate containing food particles, which was previously involved in the scene where the man is holding a black ladle in his right hand and a plate in his left hand, was located?",
#                 "answer": "Of course, if I remember correctly, the plate was on the cutting board on the table, to the right of the stove previously.",
#                 "class": "LM",
#                 "time": 275
#             },
#             {
#                 "question": "Is there any bottles on the table in the video? Please help me look for it.",
#                 "answer": "Yes, There has a bottle containing the golden-yellow liquid is on the table next to a brown or black bottle.",
#                 "class": "CS",
#                 "time": 360
#             }
#         ]
    
# time_line = [int(ques['time']) for ques in question_list]
    
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
        long_memory_list, long_memory_text_list = search_tree_multi_modal_with_embedding(long_memory_tree_cache, question, short_memory_embedding, embedding_model, embedding_tokenizer)
        # long_memory_list, long_memory_text_list = search_tree_multi_modal(long_memory_tree_cache, question_embeddings, short_memory_embedding, model, tokenizer)
        
        print("long memory list:{}".format(len(long_memory_list)))
        print("long memory text list:{}".format(len(long_memory_text_list)))
        
        long_memory_embeddings = torch.cat(long_memory_list, dim=0).view(-1, long_memory_list[0].shape[-1]) # [40x36, 4096]
        
        print("long_memory_embeddings", long_memory_embeddings.shape)
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
    update_event, 
    stop_all_thread_signal,
    start_inference_event, 
    video_reader_event,  
    summarizer_model, 
    summarizer_tokenzier, 
    building_multi_modal_memory_tree
    ):
    """
    Thread function to handle user input.
    """
    
    global short_memory_buffer
    global long_memory_tree
    global feature_bank
    global time_triger
    
    captioning = "Please describe what you see in this video in as much detail as possible from a first-person perspective, including the surrounding environment, what objects are there, etc."
    
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
    
    while not stop_all_thread_signal.is_set():
        # if not start_inference_triger and finish_infernece:
        
        # if len(feature_bank) > 20:
        if update_event.wait(0.5):
            # update_event.wait()
            video_reader_event.clear()
            print("<<<< start building memory >>>>")
            cache = feature_bank
            if len(cache) > 8:
                time_1 = time.time()
                if building_multi_modal_memory_tree:
                    print("<<<< Multi_modal Tree >>>>")
                    short_memory_buffer, long_memory_tree = long_short_memory_update_with_summarize(cache, summarizer_model, summarizer_tokenzier, captioning_input_ids,
                                                                                                    short_window=20, remember_window=10, tau=10, 
                                                                                                    compress_rate=1, chunk_size=10, 
                                                                                                    num_clusters=5, interval=10)
                else:
                    short_memory_buffer, long_memory_tree = long_short_memory_update(cache, short_window=20, 
                                                                                    remember_window=10, tau=10, 
                                                                                    compress_rate=1, chunk_size=15, 
                                                                                    num_clusters=5, interval=5)
                time_2 = time.time()
                print("<<<< memory building finish within :{} >>>>".format(time_2 - time_1))
            else:
                print("<<<< low cache mode not need long memory >>>>")
                short_memory_buffer = [cache[i] for i in range(len(cache))]
                long_memory_tree = None
                print("short_memory_buffer", len(short_memory_buffer))
            
            assert len(short_memory_buffer) > 0, "No memory ?"
            
            update_event.clear()
            start_inference_event.set()
            video_reader_event.set()
        else:
            time.sleep(0.5)
        # else:
        #     time.sleep(0.5) # avoid GIL
    print("memory thread end here !!")
    
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
    update_event,
    time_line):
    """
    Thread function to read video frames and put them into a queue.
    """
    # 两个线程之间发生一定的资源互占的情况
    
    global frame_bank
    global feature_bank
    global time_count
    global time_index
    
    current_frame_rate = 0
    # count = 0
    last_frame = None
    change_time = 0
    # time_count = 0
    # time_index = 0
    
    time_bank_1 = []
    time_bank_2 = []
    all_time_bank = []
    mag_bank = []
    
    pbar = tqdm(total=total_frames, desc="Processing Video", unit="frame")
    
    while cap.isOpened() and current_frame_rate < total_frames:
        video_reader_event.wait()
        time_1 = time.time()
        ret, frame = cap.read()
        if not ret:
            break
        
        with torch.no_grad():
            current_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # Save the last frame to image_show.jpg
            if time_count > 1:
                time_2 = time.time()
                # is_change, mean_mag, current_frame_tensor = SSIM(last_frame, current_frame, image_processor, model, 0.9) # judging by SSIM
                is_change, mean_mag, current_frame_tensor = Optical_flow(last_frame, current_frame, image_processor, model, 0.45) # judging by optical flow
                torch.cuda.empty_cache()
                time_3 = time.time()
                mag_bank.append(mean_mag)
                all_time_bank.append((time_3-time_2))
                if is_change:
                    change_time += 1
                    time_4 = time.time()
                    image_embedding = model.encode_images(current_frame_tensor.unsqueeze(0).to(dtype=torch.float16))
                    # image_embedding = model.only_encode(current_frame_tensor.unsqueeze(0).to(dtype=torch.float16))
                    # print("image embedding:{}".format(image_embedding.shape)) # 1 144 3584
                    # image_embedding = model.only_project(image_embedding)
                    time_5 = time.time()
                    # 确保有足够的帧数进行推理
                    with mutex:
                        # Enqueue frame and feature
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
                    image_embedding = model.encode_images(current_frame_tensor.unsqueeze(0).to(dtype=torch.float16))
                    # image_embedding = model.only_encode(current_frame_tensor.unsqueeze(0).to(dtype=torch.float16))
                    # print("image embedding without proj:{}".format(image_embedding.shape)) # 1 576 1024
                    # image_embedding = model.only_project(image_embedding)

                    feature_bank.append(image_embedding)
                
                # del image_embedding
                del current_frame_tensor
            
            else:
                mean_mag = 0.0
                all_time_bank.append(0.000001)
       
       
        last_frame = current_frame
        current_frame_rate += 1
        # with condition:
        time_count += 1
        if time_count in [t*frame_rate for t in time_line]:
            update_event.set()
            time_index = time_index + 1
        # condition.notify()  # 通知等待的线程
        # time_count += 1
        # time.sleep(0.01)
        time_7 = time.time()
        FPS = (time_count)/sum(all_time_bank)
        # print("FPS:{}".format(FPS))
        # Update tqdm progress bar and set postfix for FPS
        pbar.set_postfix(FPS="{:.2f}".format(FPS), MAG="{:.2f}".format(mean_mag), Time="{}".format(time_count), Buffer="{}".format(len(feature_bank)))
        pbar.update(1)
        
        if current_frame_rate == total_frames:
            stop_all_thread_signal.set()
            time.sleep(0.5)
    
    # 通过构建完整的数据
    print("stop_all_thread_signal", stop_all_thread_signal.is_set())
    
    # del feature_bank
    # del mag_bank
    for tensor in feature_bank:
        del tensor
    feature_bank.clear()

    cap.release()
    print("Video processing completed.")
    print("Find chanement {} times in {}".format(change_time, total_frames))
       
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
    update_event, 
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

        if start_inference_event.wait(0.5):
            index = time_index - 1
            update_event.clear()
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
            
            with torch.no_grad():
                print("<<<< Retrival Context >>>>")
                searched_history = build_prompt_with_search_memory_only_related(question, user_name, user_memory_index, local_memory_qa, only_related_prompt, user_keyword, ai_keyword, boot_actual_name)
                print("<<<< context retrieval finished  >>>>")
                # print("searched_history:{}".format(searched_history))

                if output_loss:
                    output_dict, _, _ = longva_inference_with_embedding_and_ppl(question, labels, num_frames, 
                                                                                    conv_mode, model,
                                                                                    tokenizer, chat, 
                                                                                    short_memory_buffer_cache, long_memory_tree_cache, searched_history)
                    loss = output_dict.loss
                    # loss = loss.detach().cpu().numpy()
                    ppl = torch.exp(loss)
                    print("loss:{}/ppl:{}".format(loss, ppl))
                    all_ppl.append(ppl.detach().cpu().numpy())
                # else:
                output, process_time, generate_time = longva_inference_with_embedding_multi_modal(question, num_frames, 
                                                                                    conv_mode, model, embedding_model,
                                                                                    tokenizer, embedding_tokenizer, chat, 
                                                                                    short_memory_buffer_cache, long_memory_tree_cache, searched_history)
            print("LongVA:", output)
            existing_data.append({"time":time_line[index],"question":question,"label":labels,"predict":output})
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
            pause_event.set()
            start_inference_event.clear()
            with open(save_file, 'w', encoding='utf-8') as file:
                json.dump(existing_data, file, ensure_ascii=False, indent=4)

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
            if len(all_ppl) == len(question_list):
                print("Avg PPL:",sum(all_ppl)/len(all_ppl))
                differences = [abs(all_ppl[i+1] - all_ppl[i]) for i in range(len(all_ppl) - 1)]
                total_difference = sum(differences)
                count_differences = len(differences)
                average_difference = total_difference / count_differences
                print("Fluency PPL:", average_difference)
                
            time.sleep(0.1)
            
    print("inference thread end here !!")
                        
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
    print("{}/{}".format(model_name, model_path))
    
    print("Initialize GPT-4o in LongVA-7B-DPO version:{} in {} mode !".format(args.conv_mode, inference_mode))
    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, None, "llava_qwen")
    
    # 1. load model
    embedding_model_id = '/13390024681/All_Model_Zoo/mxbai-colbert-large-v1'
    embedding_tokenizer = AutoTokenizer.from_pretrained(embedding_model_id)
    embedding_model = AutoModel.from_pretrained(embedding_model_id).cuda()
    
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
    
    inference_count = 0
    infernece_limit = 4
    # if inference_mode == "on_line":
    for anno in all_annotations:
        
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
                                embedding_device="cuda",
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
        # class_1 = anno['info']['class_1']
        class_2 = anno['info']['class_2']
        
        question_list = anno['breakpoint']
        time_line = [int(ques['time']) for ques in question_list]
        
        video_path = os.path.join(args.video_dir, class_2, video_name)
        
        assert os.path.exists(video_path), "{} not exist ".format(video_path)
        # if not os.path.exists(video_path):
            # break        
        # video_path = args.video_dir
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_rate = int(cap.get(cv2.CAP_PROP_FPS))
        
        current_frame = 0
        
        num_frames = args.num_frames
        print("For present model, we only support {} frames video".format(num_frames))
        # User input to control the start of model inference
        
        print("Start inference for all video !!")
        # start_inference_triger = False
        # finish_infernece = True
        
        # Create a queue to communicate with the user input thread
        pause_event = threading.Event()
        pause_event.set()  
        # Initially allow user input thread to run
        
        update_event = threading.Event()
        start_inference_event = threading.Event()
        video_reader_event = threading.Event()
        update_event.clear()
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
                                            update_event,
                                            time_line))
        video_thread.start()
        # video_thread = threading.Thread(target=video_reader_thread, args=(cap, frame_queue, total_frames, frame_rate))
        
        # Start the memory update thread
        update_thread = threading.Thread(target=updating_memory_buffer, 
                                         args=(
                                             update_event, 
                                             stop_all_thread_signal,
                                             start_inference_event, 
                                             video_reader_event, 
                                             model, 
                                             tokenizer, 
                                             args.multi_modal_memory))
        update_thread.start()
        
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
                                            update_event, 
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

        # stop_all_thread_signal.set()
        
        video_thread.join()
        
        stop_all_thread_signal.set()
        
        update_thread.join()
        infer_thread.join()
        
        global time_count 
        global time_index
        time_count = 0
        time_index = 0
        # 强制进行垃圾回收以确保资源被释放
        gc.collect()
        torch.cuda.empty_cache()
        inference_count += 1
        if inference_count > infernece_limit:
            print("infernece finished !!")
            break


if __name__ == "__main__":
    args = parse_args()
    run_inference(args)