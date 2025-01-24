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
from utiles import compute_gradients, Optical_flow

import torch.nn.functional as F

from PIL import Image
import math
import torch
import time
import cv2
import threading
import queue
import time

# torch._dynamo.config.cache_size_limit = 64
# torch._dynamo.config.suppress_errors = True
# torch.set_float32_matmul_precision('high')
# import ChatTTS
# from IPython.display import Audio
# from TTS.api import TTS

# frame_bank = []
# feature_bank = []
# frame_lock = threading.Lock()

# Global variables for shared resources
frame_bank = queue.Queue()
feature_bank = queue.Queue()
mutex = threading.Lock()

def llava_inference(video_frames, question, conv_mode, model, tokenizer, image_processor, image_sizes, chat):
    if model.config.mm_use_im_start_end:
        qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + question
    else:
        qs = DEFAULT_IMAGE_TOKEN + '\n' + question

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
    time_0 = time.time()
    image_tensor = process_images(video_frames, image_processor, model.config)  
    time_1 = time.time()
    
    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            images=image_tensor.to(dtype=torch.float16, device='cuda', non_blocking=True),
            image_sizes=image_sizes,
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
    
    print("process time:{}, generate time:{}, tts time:{} ".format((time_1 - time_0), (time_2 - time_1), (time_3 - time_2)))
    
    return outputs


def llava_inference_with_embedding(question, conv_mode, model, tokenizer, chat, feature_list):
    if model.config.mm_use_im_start_end:
        qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + question
    else:
        qs = DEFAULT_IMAGE_TOKEN + '\n' + question

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
    time_0 = time.time()
    # with threading.Lock():
    image_embeddings = topk_feature(feature_list, 4, model, question, tokenizer) # x 576 dimension
    time_1 = time.time()
    
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
    
    print("process time:{}, generate time:{}, tts time:{} ".format((time_1 - time_0), (time_2 - time_1), (time_3 - time_2)))
    
    return outputs

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

    return parser.parse_args()

def load_video_fps(vis_path, target_fps):
    """
    Load all video frames from a video file.

    Parameters:
    vis_path (str): Path to the video file.

    Returns:
    list: List of PIL.Image.Image objects representing video frames.
    """

    # Load video with VideoReader
    vr = VideoReader(vis_path, ctx=cpu(0))
    video_fps = vr.get_avg_fps()  # Get the video's frame rate
    total_frame_num = len(vr)
    
    # Calculate the interval between frames to meet the target_fps
    interval = math.ceil(video_fps / target_fps)
    
    # Get indices of frames to extract
    frame_idx = list(range(0, total_frame_num, interval))
    
    # Extract frames as numpy array
    img_array = vr.get_batch(frame_idx).asnumpy()  # T H W C

    original_size = (img_array.shape[-2], img_array.shape[-3])  # (width, height)
    original_sizes = (original_size,) * len(frame_idx)

    clip_imgs = [Image.fromarray(img_array[j]) for j in range(len(frame_idx))]

    return clip_imgs, original_sizes

def load_video_frame(frame_bank, num_frm, model):
    """
    Function to process a single video frame and prepare it for inference.
    """
    # frame bank contains all the video frame that you need 
    # Placeholder function for demonstration purposes.
    # In actual implementation, adapt this function to your needs.
    total_frames = len(frame_bank)
    # sampled_sizes = []
    if total_frames < num_frm:
        sampled_frames = [Image.fromarray(frame_bank[j]) for j in range(len(frame_bank))]
        sampled_sizes = [(frame_bank[i].shape[1], frame_bank[i].shape[0]) for i in range(len(frame_bank))]
    else:
        interval = total_frames // num_frm
        sampled_frames = [Image.fromarray(frame_bank[i * interval]) for i in range(num_frm)]
        sampled_sizes = [(frame_bank[i * interval].shape[1], frame_bank[i * interval].shape[0]) for i in range(num_frm)]
        
    return sampled_frames, sampled_sizes

def mean_absolute_error(frame1, frame2):
    return np.mean(np.abs(frame1 - frame2))

def cut_video_clip(video_path, max_change_indices, frame_indices, total_frames, fps, output_folder):
    if os.path.exists(output_folder):
        print("path already exists !!")
    else:
        os.mkdir(output_folder)
        print("create path :{}".format(output_folder))
        
    # 根据变化最大帧的索引切分视频
    print("you need to segment the video into {} parts".format(len(max_change_indices) + 1))
    cap = cv2.VideoCapture(video_path)
    for i in tqdm(range(len(max_change_indices) + 1)): # 4 + 1
        if i == 0:
            idx = max_change_indices[0]
            start_frame_idx = frame_indices[0]
            end_frame_idx = frame_indices[idx + 1] if idx < len(frame_indices) - 1 else total_frames - 1
        elif i == len(max_change_indices):
            idx = max_change_indices[i - 1]
            start_frame_idx = frame_indices[idx]
            end_frame_idx   = frame_indices[-1]
        else:
            start_idx = max_change_indices[i - 1]
            idx = max_change_indices[i]
            start_frame_idx = frame_indices[start_idx]
            end_frame_idx = frame_indices[idx + 1] if idx < len(frame_indices) - 1 else total_frames - 1
            
        # 设置视频捕获的开始和结束位置
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame_idx)
        
        # 创建视频写入对象
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 视频编解码器
        output_video_path = f'{output_folder}/segment_{i + 1}.mp4'
        out = cv2.VideoWriter(output_video_path, fourcc, fps, (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))
        
        # 读取并写入视频帧
        for frame_idx in range(start_frame_idx, end_frame_idx + 1):
            ret, frame = cap.read()
            if ret:
                out.write(frame)
        
        # 释放视频写入对象
        out.release()

def cut_video_baed_on_event(video_path):    
    # cap = cv2.VideoCapture(video_path)
    # total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    # frame_rate = int(cap.get(cv2.CAP_PROP_FPS))
    cap = cv2.VideoCapture(video_path)
    
    # 检查视频是否成功打开
    if not cap.isOpened():
        print("Error: Could not open video.")
        exit()
    
    # 获取视频总帧数
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # 设置抽帧间隔
    frame_interval = int(fps)  # 每秒抽取一帧

    # 读取第一帧
    ret, prev_frame = cap.read()
    if not ret:
        print("Error: Could not read the first frame.")
        exit()

    # 将第一帧转换为灰度图像
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

    # 存储相似度
    similarities = []

    time_1 = time.time()
    # for _ in tqdm(range(total_frames - 1), desc="Processing frames"):
    for frame_idx in tqdm(range(0, total_frames, frame_interval), desc="Processing frames"):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        
        # 读取下一帧
        ret, curr_frame = cap.read()
        if not ret:
            break

        # 将当前帧转换为灰度图像
        curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)

        # 计算相似度
        similarity = mean_absolute_error(prev_frame, curr_frame) # 前一帧减去当前帧
        similarities.append((frame_idx, similarity))

        # 当前帧变为前一帧
        prev_frame = curr_frame
        prev_gray = curr_gray

    # 释放视频捕获对象
    cap.release()
    time_2 = time.time()
    print("time spend :{}".format(time_2 - time_1))
    # 将帧号和相似度分开
    frame_indices, similarity_values = zip(*similarities)

    # # 找到变化最大的帧位置
    # max_change_index = np.argmax(similarity_values)
    # max_change_frame_idx = frame_indices[max_change_index]
    # prev_frame_idx = frame_indices[max_change_index - 1] if max_change_index > 0 else frame_indices[0]
    # after_frame_idx = frame_indices[max_change_index + 1] if max_change_index > 0 else frame_indices[0]
    # after_after_frame_idx = frame_indices[max_change_index + 2] if max_change_index > 0 else frame_indices[0]
    # print(similarity_values[max_change_index - 1], similarity_values[max_change_index], similarity_values[max_change_index + 1], similarity_values[max_change_index + 2])
    
    # 找到相似度变化最大的帧位置
    max_change_indices = np.argsort(similarity_values)[-4:]
    max_change_indices = sorted(max_change_indices)  # 保持顺序
    cut_video_clip(video_path, max_change_indices, frame_indices, total_frames, fps, "/13390024681/llama/EfficientVideo/Ours/save_videos/6")
    # pass
    return "/13390024681/llama/EfficientVideo/Ours/save_videos/6"

def load_video_frame_based_on_similarity(frame_bank, num_frm, model, image_processor, question_text, tokenizer, chunk):
    """
    Function to process a single video frame and prepare it for inference.
    """
    # 当图片的数量变得非常大的时候，整体的计算复杂度会就上去了
    # frame bank contains all the video frame that you need 
    # Placeholder function for demonstration purposes.
    # In actual implementation, adapt this function to your needs.
    total_frames = len(frame_bank)
    question_ids = tokenizer(question_text).input_ids
    question_embeddings  = model.get_model().embed_tokens(torch.tensor(question_ids, dtype=torch.long, device='cuda'))
    print("question embeddings : {}".format(question_embeddings.shape)) # num_text_token 4096
    print("There are total :{} frames in this video.".format(total_frames)) # how to speed up ?
    time_1 = time.time()
    
    concat_images = process_images(frame_bank, image_processor, model.config).to(dtype=torch.float16, device='cuda', non_blocking=True).squeeze(0)  # spend too much time 
    print("image_tensor :{}".format(concat_images.shape)) # 1 num_frame 3 336 366 
    
    # if type(frame_bank) is list:
    #     images = [x.unsqueeze(0) if x.ndim == 3 else x for x in image_tensor]
        
    # concat_images = torch.cat([image for image in images], dim=0)
    
    image_features = model.encode_images(concat_images) # num_frame 576 4096
    print("image_features shape :{}".format(image_features.shape))
    time_2 = time.time()
    simarity = image_features @ question_embeddings.permute(1, 0) # num_frame 576 num_text_token
    simarity = simarity.sum(dim=1).sum(dim=1)
    time_3 = time.time()
    print("simarity shape :{}".format(simarity.shape))
    topk_values, topk_indices = torch.topk(simarity, num_frm)
    print("time spend:{}, {}".format((time_2-time_1), (time_3-time_2))) # 3 minet ?
    for index in topk_indices:
        print(index)
        last_frame = frame_bank[index]
        
        # img = Image.fromarray(cv2.cvtColor(last_frame, cv2.COLOR_BGR2RGB))
        save_path = "./save_images/{}_rgb".format(chunk)
        if os.path.exists(save_path):
            print("path exist !")
        else:
            os.mkdir(save_path)
            
        last_frame.save("{}/image_max_{}.jpg".format(save_path, index))
    # assert 1==2

def video_reader_thread(cap, frame_queue, total_frames, frame_rate):
    """
    Thread function to read video frames and put them into a queue.
    """
    global frame_bank
    current_frame = 0
    while cap.isOpened() and current_frame < total_frames:
        ret, frame = cap.read()
        if not ret:
            break
        frame_queue.put(frame)
        # frame_bank.append(frame) # length keep increase 

        frame_bank.append(frame) # continue to increase during inference
        current_frame += frame_rate
        # Save the last frame to image_show.jpg
        last_frame = frame
        img = Image.fromarray(cv2.cvtColor(last_frame, cv2.COLOR_BGR2RGB))
        img.save("./image_show.jpg")
        time.sleep(1)  # Adjust sleep time according to frame rate
    cap.release()
    print("Video processing completed.")
    
def inference_thread(input_queue, frame_queue, model, tokenizer, image_processor, num_frames, conv_mode, pause_event, chat):
    """
    Thread function to run inference on video frames.
    """
    global frame_bank
    while True:
        if not input_queue.empty():
            question = input_queue.get()
            if question.lower() == 'exit':
                print("Exit command received. Terminating inference.")
                break
            
            # Collect frames for inference
            # while len(frame_bank) < num_frames and not frame_queue.empty():
            #     frame_bank.append(frame_queue.get())
            
            # if len(frame_bank) < num_frames:
            #     print("Not enough frames for inference {}. Waiting for more frames...".format(len(frame_bank)))
            #     time.sleep(1)
                # continue
            # Collect frames for inference
            with frame_lock:
                if len(frame_bank) < num_frames:
                    print("Not enough frames for inference. Waiting for more frames...")
                    time.sleep(1)
                    continue

            # Process the frames
            frame_bank_inf, sizes = load_video_frame(frame_bank, num_frames, model)
            
            # Run inference on the video frames and add the output to the list
            output = llava_inference(frame_bank_inf, question, conv_mode, model,
                                     tokenizer, image_processor, sizes, chat)
            print("LLaVA:", output)
            pause_event.set()

            # frame_bank   = []  # Clear frame bank for next inference

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

def queue_to_list(queue_instance):
    result_list = []
    while not queue_instance.empty():
        item = queue_instance.get()
        result_list.append(item)
    return result_list

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

    all_image_features = torch.cat(feature_list)
    print("all_image_features shape :{}".format(all_image_features.shape))
    time_2 = time.time()
    simarity = all_image_features @ question_embeddings.permute(1, 0) # num_frame 576 num_text_token
    simarity = simarity.sum(dim=1).sum(dim=1)
    time_3 = time.time()
    print("simarity shape :{}".format(simarity.shape))
    topk_values, topk_indices = torch.topk(simarity, num_frm)
    topk_features = [feature_list[idx] for idx in topk_indices]
    # Concatenate all top-k features
    concatenated_features = torch.cat(topk_features, dim=0) # num_select 576 4096
    concatenated_features = concatenated_features.view(-1, concatenated_features.shape[-1])
    print("concatenated_features shape :{}".format(concatenated_features.shape))

    return concatenated_features
 
def video_reader_thread_with_embedding(cap, total_frames, frame_rate, image_processor, model):
    """
    Thread function to read video frames and put them into a queue.
    """
    # 两个线程之间发生一定的资源互占的情况
    
    global frame_bank
    global feature_bank
    
    current_frame_rate = 0
    count = 0
    last_frame = None
    change_time = 0
    
    pbar = tqdm(total=total_frames, desc="Processing Video", unit="frame")
    
    while cap.isOpened() and current_frame_rate < total_frames:
        time_1 = time.time()
        ret, frame = cap.read()
        if not ret:
            break
        current_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # Save the last frame to image_show.jpg
        if count > 1:
            is_change, mean_mag, current_frame_tensor = Optical_flow(last_frame, current_frame, image_processor, model, 0.3) # judging by optical flow
            if is_change:
                change_time += 1
                image_embedding = model.encode_images(current_frame_tensor.unsqueeze(0).to(dtype=torch.float16, device='cuda'))
                # 确保有足够的帧数进行推理
                with threading.Lock():
                    # Enqueue frame and feature
                    frame_bank.put(current_frame)
                    feature_bank.put(image_embedding)
                    # frame_bank.append(current_frame)
                    # feature_bank.append(image_embedding)
                img = Image.fromarray(current_frame)
                img.save("/13390024681/llama/EfficientVideo/Ours/image_show.jpg")
                # mutex.release()
        else:
            mean_mag = 0.0
       
       
        last_frame = current_frame
        current_frame_rate += 1
        count += 1
        time_2 = time.time()
        FPS = 1/(time_2 - time_1)
        # print("FPS:{}".format(FPS))
        # Update tqdm progress bar and set postfix for FPS
        pbar.set_postfix(FPS="{:.2f}".format(FPS), MAG="{:.2f}".format(mean_mag))
        pbar.update(1)
        
    cap.release()
    print("Video processing completed.")
    print("Find chanement {} times in {}".format(change_time, total_frames))
    
def inference_thread_with_emebdding(input_queue, model, tokenizer, image_processor, num_frames, conv_mode, pause_event, chat):
    """
    Thread function to run inference on video frames.
    """
    global frame_bank
    global feature_bank
    
    while True:
        if not input_queue.empty():
            question = input_queue.get()
            if question.lower() == 'exit':
                print("Exit command received. Terminating inference.")
                break
            
        
            # Ensure enough frames are available for inference
            with threading.Lock():
                if feature_bank.qsize() < num_frames:
                    print("Not enough frames for inference. Waiting for more frames...")
                    time.sleep(1)
                    continue
            
            # Retrieve frames and features from queues
            with threading.Lock():
                frames = [frame_bank.get() for _ in range(num_frames)]
                features = [feature_bank.get() for _ in range(num_frames)]

            output = llava_inference_with_embedding(question, conv_mode, model,
                                    tokenizer, chat, features)
            print("LLaVA:", output)
            pause_event.set()

            # frame_bank   = []  # Clear frame bank for next inference
                   
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
    
    if inference_mode == "off_line":
        video_path = args.video_dir
        save_path = cut_video_baed_on_event(video_path)
        for index, file_path in enumerate(os.listdir(save_path)):
            video_path = os.path.join(save_path, file_path)
            video_frames, sizes = load_video_fps(video_path, target_fps=1) # 抽取所有的视频帧   
            video_frames_infer = load_video_frame_based_on_similarity(video_frames, 
                                                                    args.num_frames, 
                                                                    model, 
                                                                    image_processor,
                                                                    "Who is the pilot of the jet?",
                                                                    tokenizer,
                                                                    index)
        assert 1==2
            # Run inference on the video and add the output to the list
            # output = llava_inference(video_frames_infer, "Who is the pilot of the jet?", conv_mode, model,
            #                                 tokenizer, image_processor, sizes)
            # print(output)
    elif inference_mode == "on_line":
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
        
        # Create a queue to communicate with the user input thread
        input_queue = queue.Queue()
        
        # frame_queue = queue.Queue()
        # enter = True
        pause_event = threading.Event()
        pause_event.set()  # Initially allow user input thread to run
        # Start the user input thread
        input_thread = threading.Thread(target=user_input_thread, args=(input_queue, pause_event))
        input_thread.start()
        # print("User thread start !!")
        
        # Start the video reader thread
        # pbar = tqdm(total=total_frames, desc="Processing Video", unit="frame")
        video_thread = threading.Thread(target=video_reader_thread_with_embedding, args=(cap, total_frames, frame_rate, image_processor, model))
        # video_thread = threading.Thread(target=video_reader_thread, args=(cap, frame_queue, total_frames, frame_rate))
        video_thread.start()
        
        # Start the inference thread
        infer_thread = threading.Thread(target=inference_thread_with_emebdding, args=(input_queue, model, tokenizer, image_processor, args.num_frames, args.conv_mode, pause_event, chat))
        # infer_thread = threading.Thread(target=inference_thread, args=(input_queue, frame_queue, model, tokenizer, image_processor, args.num_frames, args.conv_mode, pause_event, chat))
        infer_thread.start()

        input_thread.join()
        video_thread.join()
        infer_thread.join()
        
    
        # if len(output_list) > sample_num:
        #     print("sample over !!")
        #     break


if __name__ == "__main__":
    args = parse_args()
    run_inference(args)