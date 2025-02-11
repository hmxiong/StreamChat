import cv2
import os
import time
from tqdm import tqdm

def extract_frames(video_path, output_folder, fps):
    # 打开视频文件
    video = cv2.VideoCapture(video_path)
    
    # 获取视频的帧率和总帧数
    video_fps = video.get(cv2.CAP_PROP_FPS)
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    print("total_frames:{} and video_fps:{}".format(total_frames, video_fps))
    # 计算帧间隔
    frame_interval = int(video_fps / fps)
    
    video_name = video_path.split("/")[-1].split(".")[0]
    save_path = os.path.join(output_folder, video_name)
    
    # 计算视频时长
    total_duration_sec = total_frames / video_fps
    hours = int(total_duration_sec // 3600)
    minutes = int((total_duration_sec % 3600) // 60)
    seconds = int(total_duration_sec % 60)
    video_duration = f"{hours:02}:{minutes:02}:{seconds:02}"
    
    print(f"Video duration: {video_duration}")
    
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    current_frame = 0
    extracted_frames = 0
    
    # with tqdm(total=total_frames, desc="Extracting frames") as pbar:
    #     while video.isOpened():
    #         ret, frame = video.read()
    #         if not ret:
    #             break
            
    #         if current_frame % frame_interval == 0:
    #             frame_filename = os.path.join(save_path, f"extracted_frames.jpg")
    #             cv2.imwrite(frame_filename, frame)
    #             extracted_frames += 1
    #             time.sleep(0.5) 
            
    #         current_frame += 1
    #         pbar.update(1)
        
    video.release()
    print(f"Extracted {extracted_frames} frames at {fps} FPS and saved to {output_folder}")

# 示例用法
video_path = ''
output_folder = ''
fps = 1  # 提取每秒1帧

extract_frames(video_path, output_folder, fps)