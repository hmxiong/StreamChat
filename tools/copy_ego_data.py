import json
import shutil
import os
import tqdm

# 设置文件路径
json_file_path = '/13390024681/llama/EfficientVideo/Ours/tools/Ego_sampled_3.json'  # 存储文件名的 JSON 文件
source_folder = '/13390024681/All_Data/EgoSchema/good_clips_git'  # 原文件夹路径
destination_folder = '/13390024681/All_Data/EgoSchema/EgoSampled_3'  # 目标文件夹路径

# 创建目标文件夹，如果不存在
if not os.path.exists(destination_folder):
    os.makedirs(destination_folder)

# 读取 JSON 文件
with open(json_file_path, 'r') as json_file:
    video_list = json.load(json_file)

# 复制文件
for video_name in tqdm.tqdm(video_list):
    source_file_path = os.path.join(source_folder, video_name)
    destination_file_path = os.path.join(destination_folder, video_name)
    if os.path.exists(source_file_path):
        shutil.copy(source_file_path, destination_file_path)
        print(f"Copied {video_name} to {destination_folder}")
    else:
        print(f"File {video_name} does not exist in {source_folder}")

print("File copying completed.")