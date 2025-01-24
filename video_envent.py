import os
import cv2
import torch
import time
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from torchvision import transforms
from decord import VideoReader, cpu

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
    pass

def mean_absolute_error_torch(frame1, frame2, device='cuda'):
    # frame1 = torch.tensor(frame1).permute(2, 0, 1).unsqueeze(0).to(device, dtype=torch.float32)
    # frame2 = torch.tensor(frame2).permute(2, 0, 1).unsqueeze(0).to(device, dtype=torch.float32)
    # transform = transforms.Compose([
    #     transforms.ToTensor(),
    #     transforms.CenterCrop((224, 224))
    # ])
    # frame1 = transform(frame1).unsqueeze(0).to(device, dtype=torch.float32)
    # frame2 = transform(frame2).unsqueeze(0).to(device, dtype=torch.float32)
    # print(frame1.shape)
    return torch.mean(np.abs(frame1 - frame2)).item()
    # return np.mean(np.abs(frame1 - frame2))

def get_most_change_frame(frame_list, num_frm):
    """
    extract most change frame based on event changement
    """
    # 计算相邻帧之间的差异
    differences = []
    prev_frame = frame_list[0]
    # for i in range(1, len(frame_list)):
    for i in tqdm(range(0, len(frame_list)), desc="Calculating differences"):
        # diff = cv2.absdiff(frame_list[i], frame_list[i - 1])
        # diff_sum = np.sum(diff)
        current_frame = frame_list[i]
        diff = mean_absolute_error(prev_frame, current_frame)
        differences.append(diff)
        prev_frame = current_frame
    
    # num_frm+1个位置
    indices = np.argsort(differences)[-num_frm-1:]
    indices = sorted(indices)
    print(indices)
    # 从每段的中间抽取一帧
    segments = []
    prev_index = 0
    for index in indices:
        middle_frame_index = (prev_index + index) // 2
        segments.append(frame_list[middle_frame_index])
        prev_index = index + 1
    # 最后一段的中间帧
    middle_frame_index = (prev_index + len(frame_list) - 1) // 2
    segments.append(frame_list[middle_frame_index])
    
    return segments, differences

def load_video_with_event_chunk(video_path, num_frm):
    """
    Load video frames from a video file and segment frame based on frame comparasion.

    Parameters:
    vis_path (str): Path to the video file.
    num_frm (int): Number of frames to extract from each clip. Defaults to 100.

    Returns:
    list: List of PIL.Image.Image objects representing video frames.
    """
    # 创建一个视频捕获对象
    cap = cv2.VideoCapture(video_path)

    # 检查视频是否成功打开
    if not cap.isOpened():
        print("Error: Could not open video.")
        return []
    
    frames = []

    # 读取视频帧
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    # 释放视频捕获对象
    cap.release()
    
    important_frames = get_most_change_frame(frames, num_frm)

    return important_frames

def save_key_frames(frames, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    for i, frame_np in enumerate(tqdm(frames, desc="Saving key frames")):
        frame_grb = cv2.cvtColor(frame_np, cv2.COLOR_BGR2RGB)
        output_path = os.path.join(output_dir, f'key_frame_{i + 1}.png')
        cv2.imwrite(output_path, frame_grb)
        print(f'Saved: {output_path}')

def read_images_in_video(video_path):
    vr = VideoReader(video_path, ctx=cpu(0))
    total_frame_num = len(vr)
    fps = round(vr.get_avg_fps())
    frame_idx = [i for i in range(0, len(vr), fps)]
    spare_frames = vr.get_batch(frame_idx).asnumpy()
    return spare_frames, frame_idx
    
def main():
    video_path = "/13390024681/llama/EfficientVideo/Ours/videos/6.mp4"
    
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

    # 找到变化最大的帧位置
    max_change_index = np.argmax(similarity_values)
    max_change_frame_idx = frame_indices[max_change_index]
    prev_frame_idx = frame_indices[max_change_index - 1] if max_change_index > 0 else frame_indices[0]
    after_frame_idx = frame_indices[max_change_index + 1] if max_change_index > 0 else frame_indices[0]
    after_after_frame_idx = frame_indices[max_change_index + 2] if max_change_index > 0 else frame_indices[0]
    print("similarity in the frames within the video:", similarity_values[max_change_index - 1], similarity_values[max_change_index], similarity_values[max_change_index + 1], similarity_values[max_change_index + 2])
    
    
    # # 重新打开视频文件，读取变化最大的两帧
    # cap = cv2.VideoCapture(video_path)
    # cap.set(cv2.CAP_PROP_POS_FRAMES, prev_frame_idx)
    # ret, prev_frame = cap.read()
    # if ret:
    #     prev_frame_path = './save_images/prev_frame.png'
    #     cv2.imwrite(prev_frame_path, prev_frame)
    
    # cap = cv2.VideoCapture(video_path)
    # cap.set(cv2.CAP_PROP_POS_FRAMES, after_frame_idx)
    # ret, after_frame = cap.read()
    # if ret:
    #     after_frame_path = './save_images/after_frame.png'
    #     cv2.imwrite(after_frame_path, after_frame)
    
    # cap = cv2.VideoCapture(video_path)
    # cap.set(cv2.CAP_PROP_POS_FRAMES, after_after_frame_idx)
    # ret, after_after_frame = cap.read()
    # if ret:
    #     after_after_frame_path = './save_images/after_after_frame.png'
    #     cv2.imwrite(after_after_frame_path, after_after_frame)
        
    # cap.set(cv2.CAP_PROP_POS_FRAMES, max_change_frame_idx)
    # ret, max_change_frame = cap.read()
    # cap.release()
    # if ret:
    #     max_change_frame_path = './save_images/max_change_frame.png'
    #     cv2.imwrite(max_change_frame_path, max_change_frame)

    # # 打印保存的帧路径
    # print(f'Saved previous frame at {prev_frame_path}')
    # print(f'Saved after frame at {after_frame_path}')
    # print(f'Saved after after frame at {after_after_frame_path}')
    # print(f'Saved max change frame at {max_change_frame_path}')
    
    # # 找到相似度变化最大的帧位置
    # max_change_indices = np.argsort(similarity_values)[-4:]
    # max_change_indices = sorted(max_change_indices)  # 保持顺序
    # cut_video_clip(video_path, max_change_indices, frame_indices, total_frames, fps, "/13390024681/llama/EfficientVideo/Ours/save_videos/6_gray")

    # 计算整体平均值
    mean_similarity = np.mean(similarity_values) # 计算的是相似度数值中的均值
    # 使用Matplotlib绘制相似度曲线
    plt.figure(figsize=(40, 12))
    plt.plot(frame_indices, similarity_values, label='Mean Absolute Error between consecutive frames')
    plt.axhline(y=mean_similarity, color='r', linestyle='--', label=f'Average Similarity (MAE) = {mean_similarity:.2f}')
    plt.axvline(x=max_change_frame_idx, color='g', linestyle='--', label=f'Max Change Frame Index = {max_change_frame_idx}')
    plt.xlabel('Frame number')
    plt.ylabel('Similarity (MAE)')
    plt.title('Frame similarity over time')
    plt.legend()
    plt.savefig('./sim.jpg')
    plt.show()

def main_2():
    video_path = "/13390024681/llama/EfficientVideo/Ours/videos/6.mp4"
    
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
    # similarities = []
    # frame_idx = []
    # frames = []
    
    # transform = transforms.Compose([
    #     transforms.ToTensor(),
    #     transforms.CenterCrop((224, 224))
    # ])

    time_1 = time.time()
    # for _ in tqdm(range(total_frames - 1), desc="Processing frames"):
    # 读取视频帧
    # for index in tqdm(range(0, total_frames, frame_interval), desc="Reading frames"): # 抽取视频的速度依然比较慢
    #     cap.set(cv2.CAP_PROP_POS_FRAMES, index)
        
    #     ret, frame = cap.read()
    #     if not ret:
    #         break
    #     frames.append(frame)
    #     frame_idx.append(index)
    frames, frame_idx = read_images_in_video(video_path)
    print("length:",len(frames), len(frame_idx), frames[0].size)
    # 释放视频捕获对象
    # cap.release()
    
    important_frames, differences = get_most_change_frame(frames, 20)

    # 释放视频捕获对象
    # cap.release()
    time_2 = time.time()
    print("time spend :{}".format(time_2 - time_1))
    
    
    save_key_frames(important_frames, "/13390024681/llama/EfficientVideo/Ours/save_images")
    
    # 计算整体平均值
    mean_similarity = np.mean(differences) # 计算的是相似度数值中的均值
    # 使用Matplotlib绘制相似度曲线
    plt.figure(figsize=(40, 12))
    plt.plot(frame_idx, differences, label='Mean Absolute Error between consecutive frames')
    plt.axhline(y=mean_similarity, color='r', linestyle='--', label=f'Average Similarity (MAE) = {mean_similarity:.2f}')
    # plt.axvline(x=max_change_frame_idx, color='g', linestyle='--', label=f'Max Change Frame Index = {max_change_frame_idx}')
    plt.xlabel('Frame number')
    plt.ylabel('Similarity (MAE)')
    plt.title('Frame similarity over time')
    plt.legend()
    plt.savefig('./sim_2.jpg')
    plt.show()
    
    
if __name__ == "__main__":
    # main_2()
    # main()
    main_2()