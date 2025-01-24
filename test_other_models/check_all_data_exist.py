import os
import json
import tqdm
import sys
sys.path.append("/13390024681/llama/EfficientVideo/Ours")

from llava.eval.model_utils import load_video


video_path_dict = {
        "/13390024681/All_Data/Streaming_final":["Cooking_show" ,"Comedy_(drama)" ,"Apple_TV"], 
        "/13390024681/All_Data/Supplement_1":["Cooking", "Metalworking"]
        }

def find_key_by_category(category, data_dict):
    for key, categories in data_dict.items():
        if category in categories:
            return key
    return None
    
if __name__ == '__main__':
    
    video_source = '/13390024681/All_Data/Streaming_Bench_v0.3'
    
    lack_data = []
    
    data_path = '/13390024681/llama/EfficientVideo/Ours/streaming_bench_v0.3.json'
    with open(data_path, 'r') as f:
        data = json.load(f)
    
    for anno in tqdm.tqdm(data):
        video_name = anno['info']['video_path']
        class_1 = anno['info']['class_1']
        class_2 = anno['info']['class_2']
        
        # if class_1 == 'Ego':
        #     file_dir = "/13390024681/All_Data/EgoSchema/good_clips_git"
        # else:
        #     file_dir = find_key_by_category(class_2, video_path_dict)
        #     file_dir = os.path.join(file_dir, class_2)
        
        if os.path.exists(os.path.join(video_source, class_1, video_name)):
            print(os.path.join(video_source, class_1, video_name))
            video_frame, video_size = load_video(os.path.join(video_source, class_1, video_name), num_frm=4)
            continue
        else:
            lack_data.append(os.path.join(video_source, class_1, video_name))
        
    if len(lack_data) > 0:
        print("missing data:{}".format(lack_data))
    else:
        print("no missing data")