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
from utiles import compute_gradients, Optical_flow, SSIM

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
mutex = threading.Lock()


def llava_inference_with_embedding(question, num_frames, conv_mode, model, tokenizer, chat, feature_list, frame_list):
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
    image_embeddings, topk_indices, topk_values = topk_feature(feature_list, num_frames, model, question, tokenizer) # x 576 dimension
    time_1 = time.time()
    print("image embedding shape:{}".format(image_embeddings.shape))
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
        if count > 1:
            time_2 = time.time()
            # is_change, mean_mag, current_frame_tensor = SSIM(last_frame, current_frame, image_processor, model, 0.9) # judging by SSIM
            is_change, mean_mag, current_frame_tensor = Optical_flow(last_frame, current_frame, image_processor, model, 0.3) # judging by optical flow
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
                    # Enqueue frame and feature
                    # frame_bank.put(current_frame)
                    # feature_bank.put(image_embedding)
                    frame_bank.append(current_frame)
                    feature_bank.append(image_embedding)
                time_6 = time.time()
                img = Image.fromarray(current_frame)
                img.save("/13390024681/llama/EfficientVideo/Ours/image_show.jpg")
                # print("time spend analyze:{}/{}/{}/{}/{}".format((time_6 - time_5), (time_5 - time_4), (time_4 - time_3), (time_3 - time_2), (time_2 - time_1)))
                time_bank_1.append((time_5 - time_4))
                time_bank_2.append((time_3 - time_2))
                all_time_bank.append((time_5 - time_4))
                if len(time_bank_1) > 100 and len(time_bank_2) > 100:
                    total_time_1 = sum(time_bank_1)
                    total_time_2 = sum(time_bank_2)
                    # count = len(time_bank_1)
                    average_time_1 = total_time_1 / len(time_bank_1)
                    average_time_2 = total_time_2 / len(time_bank_1)
                    print("avg time 1:{}/ avg time 2:{}".format(average_time_1, average_time_2))
                    print("count:{}".format(count))
                    print("max mag {} and min mag {}".format(max(mag_bank), min(mag_bank)))
                    assert 1==2
        else:
            mean_mag = 0.0
            all_time_bank.append(0.00001)
       
       
        last_frame = current_frame
        current_frame_rate += 1
        count += 1
        time_7 = time.time()
        FPS = (count)/sum(all_time_bank)
        # print("FPS:{}".format(FPS))
        # Update tqdm progress bar and set postfix for FPS
        pbar.set_postfix(FPS="{:.2f}".format(FPS), MAG="{:.2f}".format(mean_mag))
        pbar.update(1)
        
    cap.release()
    print("Video processing completed.")
    print("Find chanement {} times in {}".format(change_time, total_frames))
    
def inference_thread_with_emebdding_test(input_queue, model, tokenizer, image_processor, num_frames, conv_mode, pause_event, chat):
    """
    Thread function to run inference on video frames.
    """
    global frame_bank
    global feature_bank
    
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
        if len(frame_bank) % 40 == 0 and len(frame_bank) != 0:
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
                                                                                 feature_bank, frame_bank)
            
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
    print("All model get ready !!")
    
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
        
        # Create a queue to communicate with the user input thread
        input_queue = queue.Queue()
        pause_event = threading.Event()
        pause_event.set()  # Initially allow user input thread to run
        # Start the user input thread
        # input_thread = threading.Thread(target=user_input_thread, args=(input_queue, pause_event))
                
        # Start the video reader thread
        video_thread = threading.Thread(target=video_reader_thread_with_embedding, args=(cap, total_frames, frame_rate, image_processor, model))
        # video_thread = threading.Thread(target=video_reader_thread, args=(cap, frame_queue, total_frames, frame_rate))
        
        # Start the inference thread
        infer_thread = threading.Thread(target=inference_thread_with_emebdding_test, args=(input_queue, model, tokenizer, image_processor, args.num_frames, args.conv_mode, pause_event, chat))
        # infer_thread = threading.Thread(target=inference_thread, args=(input_queue, frame_queue, model, tokenizer, image_processor, args.num_frames, args.conv_mode, pause_event, chat))
        
        video_thread.start()
        # input_thread.start()
        infer_thread.start()

        video_thread.join()
        # input_thread.join()
        infer_thread.join()


if __name__ == "__main__":
    args = parse_args()
    run_inference(args)