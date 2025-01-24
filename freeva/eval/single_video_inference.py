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

def llava_inference(video_frames, question, conv_mode, model, tokenizer, image_processor, image_sizes):
    if model.config.mm_use_im_start_end:
        qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + question
    else:
        qs = DEFAULT_IMAGE_TOKEN + '\n' + question

    conv = conv_templates[conv_mode].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
    image_tensor = process_images(video_frames, image_processor, model.config)  

    with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                images=image_tensor.to(dtype=torch.float16, device='cuda', non_blocking=True),
                image_sizes=image_sizes,
                do_sample=False,
                temperature=0,
                top_p=None,
                num_beams=1,
                max_new_tokens=128,
                use_cache=True)

    outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
    return outputs

def parse_args():
    """
    Parse command-line arguments.
    """
    parser = argparse.ArgumentParser()

    # Define the command-line arguments
    parser.add_argument('--video_path', help='input video path', required=True)
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--conv-mode", type=str, required=False, default='video-chatgpt_v1')
    parser.add_argument("--num_frames", type=int, default=4)

    return parser.parse_args()


def run_inference(args):
    """
    Run inference

    Args:
        args: Command-line arguments.
    """

    disable_torch_init()
    model_path = os.path.expanduser(args.model_name)
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, None, model_name)

    conv_mode = args.conv_mode

    video_path = args.video_path

    video_frames, sizes = load_video(video_path, num_frm=args.num_frames)

    question = "describe the video in detail"

    try:
        # Run inference on the video and add the output to the list
        output = llava_inference(video_frames, question, conv_mode, model,
                                        tokenizer, image_processor, sizes)
        print("\n\n", output)

    except Exception as e:
        print(f"Error processing video file '{video_path}': {e}")


if __name__ == "__main__":
    args = parse_args()
    run_inference(args)
