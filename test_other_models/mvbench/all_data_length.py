import os
import io
import json

# from models.videochat2_it import VideoChat2_it
# from utils.easydict import EasyDict
import torch

from transformers import StoppingCriteria, StoppingCriteriaList
from moviepy.editor import VideoFileClip
from PIL import Image
import numpy as np
import numpy as np
import imageio
# import opencv as cv2
from decord import VideoReader, cpu
import torchvision.transforms as T
# from dataset.video_transforms import (
#     GroupNormalize, GroupScale, GroupCenterCrop, 
#     Stack, ToTorchFormatTensor
# )
from torchvision.transforms.functional import InterpolationMode
from torch.utils.data import Dataset

from torchvision import transforms

import matplotlib.pyplot as plt

from tqdm import tqdm

# from IPython.display import Video, HTML

from peft import get_peft_model, LoraConfig, TaskType
import copy

data_list = {
    "Action Sequence": ("action_sequence.json", "/13390024681/All_Data/MVBench/video/star/Charades_v1_480/", "video", True), # has start & end
    "Action Prediction": ("action_prediction.json", "/13390024681/All_Data/MVBench/video/star/Charades_v1_480/", "video", True), # has start & end
    "Action Antonym": ("action_antonym.json", "/13390024681/All_Data/MVBench/video/ssv2_video/", "video", False),
    "Fine-grained Action": ("fine_grained_action.json", "/13390024681/All_Data/MVBench/video/Moments_in_Time_Raw/videos/", "video", False),
    "Unexpected Action": ("unexpected_action.json", "/13390024681/All_Data/MVBench/video/FunQA_test/test/", "video", False),
    "Object Existence": ("object_existence.json", "/13390024681/All_Data/MVBench/video/clevrer/video_validation/", "video", False),
    "Object Interaction": ("object_interaction.json", "/13390024681/All_Data/MVBench/video/star/Charades_v1_480/", "video", True), # has start & end
    "Object Shuffle": ("object_shuffle.json", "/13390024681/All_Data/MVBench/video/perception/videos/", "video", False),
    "Moving Direction": ("moving_direction.json", "/13390024681/All_Data/MVBench/video/clevrer/video_validation/", "video", False),
    "Action Localization": ("action_localization.json", "/13390024681/All_Data/MVBench/video/sta/sta_video/", "video", True),  # has start & end
    "Scene Transition": ("scene_transition.json", "/13390024681/All_Data/MVBench/video/scene_qa/video/", "video", False),
    "Action Count": ("action_count.json", "/13390024681/All_Data/MVBench/video/perception/videos/", "video", False),
    "Moving Count": ("moving_count.json", "/13390024681/All_Data/MVBench/video/clevrer/video_validation/", "video", False),
    "Moving Attribute": ("moving_attribute.json", "/13390024681/All_Data/MVBench/video/clevrer/video_validation/", "video", False),
    "State Change": ("state_change.json", "/13390024681/All_Data/MVBench/video/perception/videos/", "video", False),
    "Fine-grained Pose": ("fine_grained_pose.json", "/13390024681/All_Data/MVBench/video/nturgbd/", "video", False),
    "Character Order": ("character_order.json", "/13390024681/All_Data/MVBench/video/perception/videos/", "video", False),
    "Egocentric Navigation": ("egocentric_navigation.json", "/13390024681/All_Data/MVBench/video/vlnqa/", "video", False),
    "Episodic Reasoning": ("episodic_reasoning.json", "/13390024681/All_Data/MVBench/video/tvqa/frames_fps3_hq/", "frame", True),  # has start & end, read frame
    "Counterfactual Inference": ("counterfactual_inference.json", "/13390024681/All_Data/MVBench/video/clevrer/video_validation/", "video", False),
}

data_dir = "/13390024681/All_Data/MVBench/json"

class MVBench_dataset(Dataset):
    def __init__(self, data_dir, data_list, num_segments=8, resolution=224):
        self.data_list = []
        for k, v in data_list.items():
            with open(os.path.join(data_dir, v[0]), 'r') as f:
                json_data = json.load(f)
            for data in json_data:
                self.data_list.append({
                    'task_type': k,
                    'prefix': v[1],
                    'data_type': v[2],
                    'bound': v[3],
                    'data': data
                })
        
        self.decord_method = {
            'video': self.read_video,
            # 'gif': self.read_gif,
            'frame': self.read_frame,
        }
        
        self.num_segments = num_segments
        
        # transform
        crop_size = resolution
        scale_size = resolution
        input_mean = [0.48145466, 0.4578275, 0.40821073]
        input_std = [0.26862954, 0.26130258, 0.27577711]
        # self.transform = T.Compose([
        #     GroupScale(int(scale_size), interpolation=InterpolationMode.BICUBIC),
        #     GroupCenterCrop(crop_size),
        #     Stack(),
        #     ToTorchFormatTensor(),
        #     GroupNormalize(input_mean, input_std) 
        # ])
    
    def __str__(self):
        len_list = {}
        option_list = {}
        for data in self.data_list:
            if data['task_type'] not in len_list:
                len_list[data['task_type']] = 0
            len_list[data['task_type']] += 1
            if data['task_type'] not in option_list:
                option_list[data['task_type']] = 0
            option_list[data['task_type']] += len(data['data']['candidates'])
        
        correct = 0
        total = 0
        res = f"There are {len(self.data_list)} videos as follow:\n"
        for k, v in len_list.items():
            correct += len_list[k]
            total += option_list[k]
            res += f"{v} for {k} ({option_list[k]} options => {len_list[k]/option_list[k]*100:.2f}%)\n"
            correct = correct + 1 / option_list[k]
        res += f"Total random accuracy: {correct/total*100:.2f}%"
        return res.rstrip()
        
    def __len__(self):
        return len(self.data_list)
    
    def get_all_video_length(self):
        from tqdm import tqdm
        total_duration_seconds = 0
        for data in tqdm(self.data_list):
            if data['data_type'] == 'video':
                video_path = os.path.join(data['prefix'], data['data']['video'])
                print(video_path)
                # vid_path = os.path.join(Eval_Video_root, vid_rela_path)
                clip = VideoFileClip(video_path)
                total_duration_seconds += clip.duration
                clip.close()
                
        total_duration_hours = total_duration_seconds / 3600
        return total_duration_hours
    
    def get_index(self, bound, fps, max_frame, first_idx=0):
        if bound:
            start, end = bound[0], bound[1]
        else:
            start, end = -100000, 100000
        start_idx = max(first_idx, round(start * fps))
        end_idx = min(round(end * fps), max_frame)
        seg_size = float(end_idx - start_idx) / self.num_segments
        frame_indices = np.array([
            int(start_idx + (seg_size / 2) + np.round(seg_size * idx))
            for idx in range(self.num_segments)
        ])
        return frame_indices
    
    def read_video(self, video_path, bound=None):
        vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
        max_frame = len(vr) - 1
        fps = float(vr.get_avg_fps())
        
        images_group = list()
        frame_indices = self.get_index(bound, fps, max_frame, first_idx=0) 
        for frame_index in frame_indices:
            img = Image.fromarray(vr[frame_index].numpy())
            images_group.append(img)
        torch_imgs = self.transform(images_group)
        return torch_imgs
    
    # def read_gif(self, video_path, bound=None, fps=25):
    #     gif = imageio.get_reader(video_path)
    #     max_frame = len(gif) - 1
        
    #     images_group = list()
    #     frame_indices = self.get_index(bound, fps, max_frame, first_idx=0) 
    #     for index, frame in enumerate(gif):
    #         if index in frame_indices:
    #             img = cv2.cvtColor(frame, cv2.COLOR_RGBA2RGB)
    #             img = Image.fromarray(img)
    #             images_group.append(img)
    #     torch_imgs = self.transform(images_group)
    #     return torch_imgs
    
    def read_frame(self, video_path, bound=None, fps=3):
        max_frame = len(os.listdir(video_path))
        images_group = list()
        frame_indices = self.get_index(bound, fps, max_frame, first_idx=1) # frame_idx starts from 1
        for frame_index in frame_indices:
            img = Image.open(os.path.join(video_path, f"{frame_index:05d}.jpg"))
            images_group.append(img)
        torch_imgs = self.transform(images_group)
        return torch_imgs

    def qa_template(self, data):
        question = f"Question: {data['question']}\n"
        question += "Options:\n"
        answer = data['answer']
        answer_idx = -1
        for idx, c in enumerate(data['candidates']):
            question += f"({chr(ord('A') + idx)}) {c}\n"
            if c == answer:
                answer_idx = idx
        question = question.rstrip()
        answer = f"({chr(ord('A') + answer_idx)}) {answer}"
        return question, answer

    def __getitem__(self, idx):
        decord_method = self.decord_method[self.data_list[idx]['data_type']]
        bound = None
        if self.data_list[idx]['bound']:
            bound = (
                self.data_list[idx]['data']['start'],
                self.data_list[idx]['data']['end'],
            )
        video_path = os.path.join(self.data_list[idx]['prefix'], self.data_list[idx]['data']['video'])
        torch_imgs = decord_method(video_path, bound)
        question, answer = self.qa_template(self.data_list[idx]['data'])
            
        return {
            'video': torch_imgs, 
            'question': question, 
            'answer': answer,
            'task_type': self.data_list[idx]['task_type']
        }
    
    
#  position embedding
num_frame = 16
resolution = 224
# new_pos_emb = get_sinusoid_encoding_table(n_position=(resolution//16)**2*num_frame, cur_frame=num_frame)
# model.vision_encoder.encoder.pos_embed = new_pos_emb

dataset = MVBench_dataset(data_dir, data_list, num_segments=num_frame, resolution=resolution)
total_duration_hours = dataset.get_all_video_length()
print(f"所有视频文件的总时长为: {total_duration_hours} 小时")