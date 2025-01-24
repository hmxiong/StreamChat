import math
import os
import argparse
import json

from tqdm import tqdm
from llava.eval.model_utils import load_video

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path

from PIL import Image
import math
import torch
import time
import cv2
import threading
import queue

def llava_inference(video_frames, question, conv_mode, model, tokenizer, image_processor, image_sizes):
    if model.config.mm_use_im_start_end:
        qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + question
    else:
        qs = DEFAULT_IMAGE_TOKEN + '\n' + question

    conv = conv_templates[conv_mode].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()
    # print(question)
    input_ids, ques_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt', question=question)
    input_ids = input_ids.unsqueeze(0).cuda()
    ques_ids = ques_ids.unsqueeze(0).cuda()
    # print("input",input_ids)
    # print("ques",ques_ids)
    
    image_tensor = process_images(video_frames, image_processor, model.config)  

    with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                images=image_tensor.to(dtype=torch.float16, device='cuda', non_blocking=True),
                image_sizes=image_sizes,
                question_ids=ques_ids,
                do_sample=True if args.temperature > 0 else False,
                temperature=args.temperature,
                top_p=args.top_p,
                num_beams=args.num_beams,
                max_new_tokens=128,
                use_cache=True)

    outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
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
    parser.add_argument("--num_chunks", type=int, default=1)
    parser.add_argument("--chunk_idx", type=int, default=0)
    parser.add_argument("--num_frames", type=int, default=4)
    parser.add_argument("--device", type=str, required=False, default='cuda:0')
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=None)

    return parser.parse_args()

def load_video_frame(frame_bank, num_frm):
    """
    Function to process a single video frame and prepare it for inference.
    """
    # Placeholder function for demonstration purposes.
    # In actual implementation, adapt this function to your needs.
    total_frames = len(frame_bank)
    original_size = (frame_bank[0].shape[1], frame_bank[0].shape[0])
    original_sizes = (original_size,) * num_frm
    
    if total_frames < num_frm:
        sampled_frames = [Image.fromarray(frame_bank[j]) for j in range(len(frame_bank))]
    else:
        interval = total_frames // num_frm
        sampled_frames = [Image.fromarray(frame_bank[i * interval]) for i in range(num_frm)]
    return sampled_frames, original_sizes

def user_input_thread(input_queue, pause_event, enter_instruction):
    """
    Thread function to handle user input.
    """
    while True:
        pause_event.wait()
        # if enter_instruction:
        question = input("User Instruction: ")
        input_queue.put(question)
        if question.lower() == 'exit':
            break
        # Pause the user input thread
        pause_event.clear()
            
def run_inference(args):
    """
    Run inference on Video QA DataSetå.

    Args:
        args: Command-line arguments.
    """

    disable_torch_init()
    model_path = os.path.expanduser(args.model_name)
    model_name = get_model_name_from_path(model_path)
    print("Initialize GPT-4o!")
    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, args.model_base, model_name)

    output_list = []  # List to store the output results
    conv_mode = args.conv_mode

    video_formats = ['.mp4', '.avi', '.mov', '.mkv']

    # Iterate over each sample in the ground truth file
    index = 0
    sample_num = 100
    
    video_dir = args.video_dir
    if "msvd" in video_dir:
        mode = "MSVD"
    elif "MSRVTT" in args.video_dir:
        mode = "MSRVTT"
    elif "ActiveNet" in args.video_dir:
        mode = "ActiveNet"
    else:
        mode = "Others"
    
    # try:
    #     question = input(f"User Instruction: ")
    # except EOFError:
    #     question = ""
    # if not question:
    #     print("exit...")
        
    video_path = args.video_dir
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_rate = int(cap.get(cv2.CAP_PROP_FPS))
        
    # video_path = args.video_dir
    # video_frames, sizes = load_video(video_path, num_frm=args.num_frames) # 实际上只抽取了num_frames对应的帧数                
    # Run inference on the video and add the output to the list
    # output = llava_inference(video_frames, question, conv_mode, model,
    #                                 tokenizer, image_processor, sizes)
    # print(output)
    current_frame = 0
    num_frames = args.num_frames # 实际的流式测试我发现都非常消耗时间
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
    global enter_instruction
    enter_instruction = True
    pause_event = threading.Event()
    pause_event.set()  # Initially allow user input thread to run
    # Start the user input thread
    input_thread = threading.Thread(target=user_input_thread, args=(input_queue, pause_event, enter_instruction))
    input_thread.start()
    
    frame_bank = []
    
    while cap.isOpened() and current_frame < total_frames:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_bank.append(frame) # please tell me the time limit of this operation
        last_frame = frame
        img = Image.fromarray(cv2.cvtColor(last_frame, cv2.COLOR_BGR2RGB))
        img.save("./image_show.jpg")
        # Check for user input
        # Check for user input
        if not input_queue.empty():
            question = input_queue.get()
            if question.lower() == 'exit':
                print("Exit command received. Terminating inference.")
                break

            # Process the frame
            video_frames, sizes = load_video_frame(frame_bank, num_frm=num_frames)

            # Run inference on the video frame and add the output to the list
            output = llava_inference(video_frames, question, conv_mode, model,
                                     tokenizer, image_processor, sizes) # 整体程序容易卡死在这里
            print("LLaVA:",output)
            # Resume the user input thread
            pause_event.set()
            # enter_instruction = True
            
        current_frame += frame_rate
        time.sleep(1)

    cap.release()
    print("Video processing completed.")



if __name__ == "__main__":
    args = parse_args()
    run_inference(args)