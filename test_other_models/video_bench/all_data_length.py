import json
import argparse
import os
import tqdm
from moviepy.editor import VideoFileClip


def calculate_total_video_duration(json_file_path):
    total_duration_seconds = 0

    with open(json_file_path, 'r') as f:
        video_paths = json.load(f)

    for path in video_paths:
        clip = VideoFileClip(path)
        total_duration_seconds += clip.duration
        clip.close()

    total_duration_hours = total_duration_seconds / 3600

    return total_duration_hours

# parser = argparse.ArgumentParser()
# parser.add_argument("--dataset_name", type=str, default=None, help="The type of LLM")
# parser.add_argument("--Eval_QA_root", type=str, default='./', help="folder containing QA JSON files")
# parser.add_argument("--Eval_Video_root", type=str, default='./', help="folder containing video data")
# parser.add_argument("--chat_conversation_output_folder", type=str, default='./Chat_results', help="")
# args = parser.parse_args()

Eval_QA_root = "/13390024681/llama/EfficientVideo/Ours/test_other_models/video_bench/Video-Bench"
Eval_Video_root = "/13390024681/All_Data/Video-Bench"
dataset_qajson = {
  "Ucfcrime": f"{Eval_QA_root}/Eval_QA/Ucfcrime_QA_new.json",
  "Youcook2": f"{Eval_QA_root}/Eval_QA/Youcook2_QA_new.json",
  "TVQA": f"{Eval_QA_root}/Eval_QA/TVQA_QA_new.json",
  "MSVD": f"{Eval_QA_root}/Eval_QA/MSVD_QA_new.json",
  "MSRVTT": f"{Eval_QA_root}/Eval_QA/MSRVTT_QA_new.json",
  "Driving-decision-making": f"{Eval_QA_root}/Eval_QA/Driving-decision-making_QA_new.json",
  "NBA": f"{Eval_QA_root}/Eval_QA/NBA_QA_new.json",
  "SQA3D": f"{Eval_QA_root}/Eval_QA/SQA3D_QA_new.json",
  "Driving-exam": f"{Eval_QA_root}/Eval_QA/Driving-exam_QA_new.json",
  "MV": f"{Eval_QA_root}/Eval_QA/MV_QA_new.json",
  "MOT": f"{Eval_QA_root}/Eval_QA/MOT_QA_new.json",
  "ActivityNet": f"{Eval_QA_root}/Eval_QA/ActivityNet_QA_new.json",
  "TGIF": f"{Eval_QA_root}/Eval_QA/TGIF_QA_new.json"
}

dataset_name_list = list(dataset_qajson.keys())
print(dataset_name_list)

# os.makedirs(args.chat_conversation_output_folder, exist_ok=True)
total_duration_seconds = 0
video_count = 0
count = 0 

for dataset_name in tqdm.tqdm(dataset_name_list):
    qa_json = dataset_qajson[dataset_name]
    print(f'Dataset name:{dataset_name}, {qa_json=}!')
    with open(qa_json, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    eval_dict = {}
    for q_id, item in tqdm.tqdm(data.items()):
        # try:   
        # video_id = item['video_id']
        # question = item['question'] 
    
        # vid_rela_path = item['vid_path']
        # vid_path = os.path.join(Eval_Video_root, vid_rela_path)
        
        # clip = VideoFileClip(vid_path)
        # total_duration_seconds += clip.duration
        count += 1
print(count)
# total_duration_hours = total_duration_seconds / 3600  
# avg_hour = total_duration_hours / count
# print(f"所有视频文件的总时长为: {total_duration_hours} 小时")
# print(f"所有视频文件的平均时长为: {avg_hour} 小时")

# if __name__ == "__main__":
#     json_file_path = "your_video_paths.json"  # 替换为实际的JSON文件路径
#     total_hours = calculate_total_video_duration(json_file_path)
#     print(f"所有视频文件的总时长为: {total_hours} 小时")