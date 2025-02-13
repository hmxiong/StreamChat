
import torch
import copy, os, random
import cv2
import decord
from decord import VideoReader, cpu
decord.bridge.set_bridge('torch')
import numpy as np
from PIL import Image
from torchvision import transforms
from transformers import ProcessorMixin, BatchEncoding
from transformers.image_processing_utils import BatchFeature
from pytorchvideo.data.encoded_video import EncodedVideo
from torchvision.transforms import Compose, Lambda, ToTensor
from torchvision.transforms._transforms_video import NormalizeVideo, RandomCropVideo, RandomHorizontalFlipVideo, CenterCropVideo
from pytorchvideo.transforms import ApplyTransformToKey, ShortSideScale, UniformTemporalSubsample


OPENAI_DATASET_MEAN = (0.48145466, 0.4578275, 0.40821073)
OPENAI_DATASET_STD = (0.26862954, 0.26130258, 0.27577711)

def get_image(image_path):
    image = Image.open(image_path).convert('RGB')
    return image

def load_frames(frames_dir):
    results = []
    frame_names = os.listdir(frames_dir)
    frame_names.sort()
    # frame_files = [x for x in os.listdir(frames_dir) if x.endswith('jpg')]
    # # sort frame by name, converted to int
    # frame_files = sorted(frame_files, key=lambda x: int(x.split('.')[0]))
    for frame_name in frame_names:
        image_path = f"{frames_dir}/{frame_name}"
        results.append(image_path)
    return results

# def sample_frames(frames, num_segments):
#     if len(frames) <= num_segments:
#         return frames
#     frame_indices = list(range(len(frames)))
#     cand_indices = copy.deepcopy(frame_indices)
#     intervals = np.linspace(start=0, stop=len(frame_indices), num=num_segments + 1).astype(int)
#     ranges = []

#     for idx, interv in enumerate(intervals[:-1]):
#         ranges.append((interv, intervals[idx + 1] - 1))

#     try:
#         frame_indices = [cand_indices[random.choice(range(x[0], x[1]))] for x in ranges]
#     except:
#         frame_indices = [cand_indices[x[0]] for x in ranges]

#     sampled_frames = [frames[indice] for indice in frame_indices]

#     return sampled_frames

def sample_frames(frames, num_segments):
    duration = len(frames)
    frame_id_array = np.linspace(0, duration-1, num_segments, dtype=int)
    frame_id_list = frame_id_array.tolist()

    sampled_frames = []
    for frame_idx in frame_id_list:
        image_path = frames[frame_idx]
        image = get_image(image_path)
        sampled_frames.append(image)
    return sampled_frames

def make_list_of_images(x):
    if not isinstance(x, list):
        return [x]
    return x

def mean_absolute_error(frame1, frame2):
    return np.mean(np.abs(frame1 - frame2))

def get_most_change_frame(frame_list, num_frm):
    """
    extract most change frame based on event changement
    """
    # 计算相邻帧之间的差异
    differences = []
    prev_frame = frame_list[0]
    # for i in range(1, len(frame_list)):
    # for i in tqdm(range(0, len(frame_list)), desc="Calculating differences"):
    for i in range(0, len(frame_list)):
        # diff = cv2.absdiff(frame_list[i], frame_list[i - 1])
        # diff_sum = np.sum(diff)
        current_frame = frame_list[i]
        diff = mean_absolute_error(prev_frame, current_frame)
        differences.append(diff)
        prev_frame = current_frame
    
    # num_frm+1个位置
    indices = np.argsort(differences)[-num_frm-1:]
    indices = sorted(indices)
    # print(indices)
    # 从每段的中间抽取一帧
    segments = []
    prev_index = 0
    for index in indices:
        middle_frame_index = (prev_index + index) // 2
        segments.append(torch.from_numpy(frame_list[middle_frame_index]).permute(2, 0, 1))
        prev_index = index + 1
    # 最后一段的中间帧
    middle_frame_index = (prev_index + len(frame_list) - 1) // 2
    segments.append(torch.from_numpy(frame_list[middle_frame_index]).permute(2, 0, 1)) # c h w
    # segments.append(frame_list[middle_frame_index].permute(2, 0, 1)) # c h w
    
    return segments, differences

def get_video_transform(video_decode_backend, num_frames=8):
    if video_decode_backend == 'pytorchvideo':
        transform = ApplyTransformToKey(
            key="video",
            transform=Compose(
                [
                    UniformTemporalSubsample(num_frames),
                    Lambda(lambda x: x / 255.0),
                    NormalizeVideo(mean=OPENAI_DATASET_MEAN, std=OPENAI_DATASET_STD),
                    ShortSideScale(size=224),
                    CenterCropVideo(224),
                    RandomHorizontalFlipVideo(p=0.5),
                ]
            ),
        )
    elif video_decode_backend == 'decord':
        transform = Compose(
            [
                # UniformTemporalSubsample(num_frames),
                Lambda(lambda x: x / 255.0),
                NormalizeVideo(mean=OPENAI_DATASET_MEAN, std=OPENAI_DATASET_STD),
                ShortSideScale(size=224),
                CenterCropVideo(224),
                # RandomHorizontalFlipVideo(p=0.5),
            ]
        )

    elif video_decode_backend == 'opencv':
        transform = Compose(
            [
                # UniformTemporalSubsample(num_frames),
                Lambda(lambda x: x / 255.0),
                NormalizeVideo(mean=OPENAI_DATASET_MEAN, std=OPENAI_DATASET_STD),
                ShortSideScale(size=224),
                CenterCropVideo(224),
                # RandomHorizontalFlipVideo(p=0.5),
            ]
        )
    elif video_decode_backend == 'search':
        transform = Compose(
            [
                # UniformTemporalSubsample(num_frames),
                Lambda(lambda x: x / 255.0),
                NormalizeVideo(mean=OPENAI_DATASET_MEAN, std=OPENAI_DATASET_STD),
                ShortSideScale(size=224),
                CenterCropVideo(224),
                # RandomHorizontalFlipVideo(p=0.5),
            ]
        )
    elif video_decode_backend == 'frames':
        transform = Compose(
            [
                NormalizeVideo(mean=OPENAI_DATASET_MEAN, std=OPENAI_DATASET_STD),
                ShortSideScale(size=224),
                CenterCropVideo(224),
                # RandomHorizontalFlipVideo(p=0.5),
            ]
        )
    else:
        raise NameError('video_decode_backend should specify in (pytorchvideo, decord, opencv)')
    return transform


def load_and_transform_video(
        video_path,
        transform,
        video_decode_backend='opencv',
        clip_start_sec=0.0,
        clip_end_sec=None,
        num_frames=8,
):
    if video_decode_backend == 'pytorchvideo':
        #  decord pyav
        video = EncodedVideo.from_path(video_path, decoder="decord", decode_audio=False)
        duration = video.duration
        start_sec = clip_start_sec  # secs
        end_sec = clip_end_sec if clip_end_sec is not None else duration  # secs
        video_data = video.get_clip(start_sec=start_sec, end_sec=end_sec)
        video_outputs = transform(video_data)
    
    elif video_decode_backend == 'decord':
        decord.bridge.set_bridge('torch')
        decord_vr = VideoReader(video_path, ctx=cpu(0))
        ori_duration = len(decord_vr)
        # frame_id_list = np.linspace(0, duration-1, num_frames, dtype=int)
        fps_vid = decord_vr.get_avg_fps()
        valid_duration = min(int(fps_vid * 10), ori_duration)
        frame_id_list = np.linspace(0, valid_duration-1, num_frames, dtype=int)
        video_data = decord_vr.get_batch(frame_id_list)
        video_data = video_data.permute(3, 0, 1, 2)  # (T, H, W, C) -> (C, T, H, W)
        video_outputs = transform(video_data)

    # elif video_decode_backend == 'decord':
    #     decord.bridge.set_bridge('torch')
    #     decord_vr = VideoReader(video_path, ctx=cpu(0))
    #     duration = len(decord_vr)
    #     frame_id_list = np.linspace(0, duration-1, num_frames, dtype=int)
    #     video_data = decord_vr.get_batch(frame_id_list)
    #     video_data = video_data.permute(3, 0, 1, 2)  # (T, H, W, C) -> (C, T, H, W)
    #     video_outputs = transform(video_data)
    
    elif video_decode_backend == 'frames':
        frames = load_frames(video_path)
        frames = sample_frames(frames, num_frames)
        to_tensor = ToTensor()
        video_data = torch.stack([to_tensor(_) for _ in frames]).permute(1, 0, 2, 3) # (T, C, H, W) -> (C, T, H, W)
        video_outputs = transform(video_data)

    elif video_decode_backend == 'opencv':
        cv2_vr = cv2.VideoCapture(video_path)
        duration = int(cv2_vr.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_id_list = np.linspace(0, duration-1, num_frames, dtype=int)
        # frame_id_list = np.linspace(0, duration-5, num_frames, dtype=int)

        video_data = []
        for frame_idx in frame_id_list:
            cv2_vr.set(1, frame_idx)
            ret, frame = cv2_vr.read()
            if not ret:
                raise ValueError(f'video error at {video_path} for frame {frame_idx}')
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            video_data.append(torch.from_numpy(frame).permute(2, 0, 1))
        cv2_vr.release()
        video_data = torch.stack(video_data, dim=1)
        video_outputs = transform(video_data)
        
    elif video_decode_backend == 'search':
        decord.bridge.set_bridge('torch')
        vr = VideoReader(video_path, ctx=cpu(0))
        total_frame_num = len(vr)
        fps = round(vr.get_avg_fps())
        # frame_idx = [i for i in range(0, len(vr), fps)]
        frame_idx = [i for i in range(0, len(vr))]
        spare_frames = vr.get_batch(frame_idx).numpy() # already tensor de
        # original_size = (spare_frames.shape[-2], spare_frames.shape[-3])  # (width, height)
        # original_sizes = (original_size,) * num_frames
        video_frames, differences = get_most_change_frame(spare_frames, num_frames-2)
        # print(video_frames[0].shape)
        video_data = torch.stack(video_frames, dim=1) # c T H W 
        video_outputs = transform(video_data)
        
    else:
        raise NameError('video_decode_backend should specify in (pytorchvideo, decord, opencv)')
    return video_outputs

class LanguageBindVideoProcessor(ProcessorMixin):
    attributes = []
    tokenizer_class = ("LanguageBindVideoTokenizer")

    def __init__(self, config, tokenizer=None, **kwargs):
        super().__init__(**kwargs)
        self.config = config
        self.config.vision_config.video_decode_backend = 'search'
        video_decode_backend = self.config.vision_config.video_decode_backend
        num_frames = self.config.vision_config.num_frames
        self.transform = get_video_transform(video_decode_backend, num_frames)
        self.image_processor = load_and_transform_video
        self.tokenizer = tokenizer
        

    def __call__(self, images=None, text=None, context_length=77, return_tensors=None, **kwargs):
        if text is None and images is None:
            raise ValueError("You have to specify either text or images. Both cannot be none.")
        if text is not None:
            encoding = self.tokenizer(text, max_length=context_length, padding='max_length',
                                      truncation=True, return_tensors=return_tensors, **kwargs)

        if images is not None:
            if 'video_decode_backend' in kwargs:
                video_decode_backend = kwargs['video_decode_backend']
                num_frames = kwargs.get('num_frames', 8)
                transform_function = get_video_transform(video_decode_backend, num_frames)
            else:
                video_decode_backend = self.config.vision_config.video_decode_backend
                transform_function = self.transform
            images = make_list_of_images(images)
            image_features = [self.image_processor(image, transform_function,
                                                   video_decode_backend=video_decode_backend,
                                                   num_frames=self.config.vision_config.num_frames) for image in images]
            # image_features = [torch.rand(3, 8, 224, 224) for image in images]
            image_features = torch.stack(image_features)
        if text is not None and images is not None:
            encoding["pixel_values"] = image_features
            return encoding
        elif text is not None:
            return encoding
        else:
            return {"pixel_values": image_features}

    def preprocess(self, images, return_tensors):
        return self.__call__(images=images, return_tensors=return_tensors)

    def batch_decode(self, skip_special_tokens=True, *args, **kwargs):
        """
        This method forwards all its arguments to CLIPTokenizerFast's [`~PreTrainedTokenizer.batch_decode`]. Please
        refer to the docstring of this method for more information.
        """
        return self.tokenizer.batch_decode(*args, skip_special_tokens=skip_special_tokens, **kwargs)

    def decode(self, skip_special_tokens=True, *args, **kwargs):
        """
        This method forwards all its arguments to CLIPTokenizerFast's [`~PreTrainedTokenizer.decode`]. Please refer to
        the docstring of this method for more information.
        """
        return self.tokenizer.decode(*args, skip_special_tokens=skip_special_tokens, **kwargs)
