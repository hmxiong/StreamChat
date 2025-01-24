import os
import numpy as np
import torch
import argparse
import math
import json
import torchvision.transforms as T
from tqdm import tqdm
from decord import VideoReader, cpu
from PIL import Image, ImageFile, ImageDraw, ImageFont
from torchvision.transforms.functional import InterpolationMode
# import torchvision.transforms as transforms
from transformers import AutoModel, AutoTokenizer
import shutil
import random
from urllib.request import urlopen
# from PIL import Image, ImageFile, ImageDraw, ImageFont
 
def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]

def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]

# def get_context_emb(state, img_list):
#     prompt = state.get_prompt()
#     print(prompt)
#     prompt_segs = prompt.split('<Img><ImageHere></Img>')

#     assert len(prompt_segs) == len(
#         img_list
#     ) + 1, "Unmatched numbers of image placeholders and images."
#     seg_tokens = [
#         chat_model.tokenizer(seg, return_tensors="pt",  add_special_tokens=i == 0).input_ids.to(0)
#         for i, seg in enumerate(prompt_segs)
#     ]
#     seg_embs = [chat_model.model.tok_embeddings(seg_t) for seg_t in seg_tokens]
#     txt_mask = [torch.zeros(seg_e.shape[:2]) for seg_e in seg_embs]
#     mixed_embs = [emb for pair in zip(seg_embs[:-1], img_list) for emb in pair] + [seg_embs[-1]]
#     maxed_masks = [emb for pair in zip(txt_mask[:-1], [torch.ones(img.shape[:2]) for img in img_list]) for emb in pair] + [txt_mask[-1]]
#     mixed_embs = torch.cat(mixed_embs, dim=1)
#     maxed_masks = torch.cat(maxed_masks, dim=1).bool()
#     return mixed_embs, maxed_masks
def padding_336(b, R=336):
    width, height = b.size
    tar = int(np.ceil(height / R) * R)
    top_padding = int((tar - height) / 2)
    bottom_padding = tar - height - top_padding
    left_padding = 0
    right_padding = 0
    b = T.Pad(b, [left_padding, top_padding, right_padding, bottom_padding], fill=[255, 255, 255])

    return b

def R560_HD18_Identity_transform(img):
    width, height = img.size
    trans = False
    if width < height:
        img = img.transpose(Image.TRANSPOSE)
        trans = True
        width, height = img.size
    ratio = (width / height)
    scale = 1
    while scale * np.ceil(scale / ratio) <= 18:
        scale += 1
    scale -= 1

    scale_low = min(np.ceil(width * 1.5 / 560), scale)
    scale_up = min(np.ceil(width * 1.5 / 560), scale)
    scale = random.randrange(scale_low, scale_up + 1)

    new_w = int(scale * 560)
    new_h = int(new_w / ratio)

    img = T.Resize(img, [new_h, new_w], )
    img = padding_336(img, 560)
    width, height = img.size
    if trans:
        img = img.transpose(Image.TRANSPOSE)

    return img

def video_img_process(imgs):
    new_imgs = []
    for img in imgs:
        w, h = img.size
        scale = w / h
        if w > h:
            new_w = 560 * 2
            new_h = int(560 * 2 / scale)
        else:
            new_w = int(560 * 2 * scale)
            new_h = 560 * 2
        img = T.Resize(img, [new_h, new_w], )
        new_imgs.append(img)
    imgs = new_imgs
    new_w = 0
    new_h = 0
    pad = 40
    if w > h:
        for im in imgs:
            w, h = im.size
            new_w = max(new_w, w)
            new_h += h + 10 + pad
        truetype_url = 'https://huggingface.co/internlm/internlm-xcomposer2d5-7b/resolve/main/SimHei.ttf?download=true'
        ff = urlopen(truetype_url)
        font = ImageFont.truetype(ff, pad)
        new_img = Image.new('RGB', (new_w, new_h), 'white')
        draw = ImageDraw.Draw(new_img)
        curr_h = 0
        for idx, im in enumerate(imgs):
            w, h = im.size
            new_img.paste(im, (0, pad + curr_h))
            draw.text((0, curr_h), f'<IMAGE {idx}>', font=font, fill='black')
            if idx + 1 < len(imgs):
                draw.line([(0, pad + curr_h + h + 5), (new_w, pad + curr_h + h + 5)], fill='black', width=2)
            curr_h += h + 10 + pad
        # print (new_w, new_h)
    else:
        for im in imgs:
            w, h = im.size
            new_w += w + 10
            new_h = max(new_h, h)
        new_h += pad
        font = ImageFont.truetype("SimHei.ttf", pad)
        new_img = Image.new('RGB', (new_w, new_h), 'white')
        draw = ImageDraw.Draw(new_img)
        curr_w = 0
        for idx, im in enumerate(imgs):
            w, h = im.size
            new_img.paste(im, (curr_w, pad))
            draw.text((curr_w, 0), f'<IMAGE {idx}>', font=font, fill='black')
            if idx + 1 < len(imgs):
                draw.line([(curr_w + w + 5, 0), (curr_w + w + 5, new_h)], fill='black', width=2)
            curr_w += w + 10
    return new_img

def load_video(vis_path, num_frm=32, start=None, end=None):
    vid = VideoReader(vis_path, num_threads=1)
    fps = vid.get_avg_fps()
    t_stride = int(2 * round(float(fps) / int(1)))
    start_idx = 0 if start is None else start
    end_idx = len(vid) if end is None else end
    all_pos = list(range(start_idx, end_idx, t_stride))
    images = [vid[i].asnumpy() for i in all_pos]
    if len(images) > num_frm:
        num_frm = min(num_frm, len(images))
        step_size = len(images) / (num_frm + 1)
        indices = [int(i * step_size) for i in range(num_frm)]
        images = [images[i] for i in indices]
    images = [Image.fromarray(arr) for arr in images]
    print(f'sample {len(images)} frames.')
    img = video_img_process(images)
    return img

def img2emb(model, image):
    img_embeds, img_split = model.vit([image], 
        model.plora_glb_GN, model.plora_sub_GN)
    if len(img_split) > 1:
        print ('Batch Size >1 is not supported.')
        assert 0
    #print (img_embeds.shape)
    img_embeds = model.vision_proj(img_embeds)
    atts_img = torch.ones(
        img_embeds.size()[:-1], dtype=torch.long).to(img_embeds.device)

    img_target = torch.ones(
        img_embeds.size()[:2], dtype=torch.long).to(
            img_embeds.device) * -100

    return img_embeds, atts_img, img_target
    
def parse_args():
    """
    Parse command-line arguments.
    """
    parser = argparse.ArgumentParser()

    # Define the command-line arguments
    parser.add_argument('--model_path', help='', required=True)
    parser.add_argument('--video_dir', help='Directory containing video files.', required=True)
    parser.add_argument('--gt_file_question', help='Path to the ground truth file containing question.', required=True)
    parser.add_argument('--output_dir', help='Directory to save the model results JSON.', required=True)
    parser.add_argument('--output_name', help='Name of the file for storing results JSON.', required=True)
    parser.add_argument("--num_chunks", type=int, default=1)
    parser.add_argument("--chunk_idx", type=int, default=0)
    parser.add_argument("--device", type=str, required=False, default='cuda:0')
    parser.add_argument('--model_base', help='', default=None, type=str, required=False)
    parser.add_argument("--model_max_length", type=int, required=False, default=2048)

    return parser.parse_args()

def run_inference(args):
    # If you have an 80G A100 GPU, you can put the entire model on a single GPU.
    # Otherwise, you need to load a model using multiple GPUs, please refer to the `Multiple GPUs` section.
    # print("copy file from another direction ")
    
    # shutil.copy("/13390024681/llama/EfficientVideo/Ours/ixc_utils.py", "/root/.cache/huggingface/modules/transformers_modules/internlm-xcomposer2d5-7b/ixc_utils.py")
    torch.set_grad_enabled(False)
    print("Init model from {}".format(args.model_path))
    path = args.model_path
    model = AutoModel.from_pretrained(
        path,
        torch_dtype=torch.bfloat16,
        # device_map='cuda',
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        local_files_only=True
        ).eval().cuda().half()
    tokenizer = AutoTokenizer.from_pretrained(
        path, 
        trust_remote_code=True, 
        use_fast=False)
    model.tokenizer = tokenizer
    
    vis_processor = T.Compose([
            T.ToTensor(),
            T.Normalize((0.48145466, 0.4578275, 0.40821073),
                        (0.26862954, 0.26130258, 0.27577711)),
        ])
    
    # prepare for generation
    gt_questions = json.load(open(args.gt_file_question, "r"))
    gt_questions = get_chunk(gt_questions, args.num_chunks, args.chunk_idx)

    answers_file = os.path.join(args.output_dir, f"{args.output_name}.json")
    os.makedirs(args.output_dir, exist_ok=True)
    ans_file = open(answers_file, "w")

    # Create the output directory if it doesn't exist
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    output_list = []  # List to store the output results

    video_dir = args.video_dir

    # Iterate over each sample in the ground truth file
    index = 0
    for sample in tqdm(gt_questions, desc="InterLM_XCP Inference for StreamBench"):
        video_name = sample['info']['video_path']
        class_1 = sample['info']['class_1']
        class_2 = sample['info']['class_2']
        
        # Load the video file
        # for fmt in video_formats:  # Added this line
        temp_path = os.path.join(video_dir, class_1, f"{video_name}")
        if os.path.exists(temp_path):
            video_path = temp_path
            image = video_path
        
        for ques_sample in sample['breakpoint']:
            
            question = ques_sample['question']
            answer = ques_sample['answer']
            id = ques_sample['time']
            qa_class = ques_sample['class']
            
            index += 1

            sample_set = {'id': id, 'question': question, 'answer': answer, 'class': qa_class}
            
            img_pil = load_video(image)
            img_pil = R560_HD18_Identity_transform(img_pil)
            # state.single = True
            img_str = "<Img><ImageHere></Img>"

            img = vis_processor(img_pil).unsqueeze(0).to(devide=model.device, dtype=torch.bfloat16)
            
            # with torch.autocast():
            # # device_type='cuda', dtype=torch.float16
            #     with torch.no_grad():
            #         response, _ = model.chat(tokenizer, question, image, do_sample=False, use_meta=True)
            # print(response)
            print(f'User: {question}\nAssistant: {response}')
            sample_set['pred'] = response
                
            output_list.append(sample_set)
            ans_file.write(json.dumps(sample_set) + "\n")
            
    ans_file.close()


if __name__ == "__main__":
    args = parse_args()
    run_inference(args)