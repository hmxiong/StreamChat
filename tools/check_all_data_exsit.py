import os
import json
import tqdm

if __name__ == '__main__':
    
    video_path_dict = {
        "/13390024681/All_Data/Streaming_final":["Cooking_show" ,"Comedy_drama" ,"Apple_TV"], 
        "/13390024681/All_Data/Supplement_1":["Cooking", "Metalworking"]
        }

    def find_key_by_category(category, data_dict):
        for key, categories in data_dict.items():
            if category in categories:
                return key
        return None
    
    file_dir = '/13390024681/All_Data/ActNet-Video/all_test'
    data_path ='/13390024681/All_Data/ActiveNet-json/actnet_test.json'
    with open(data_path, 'r') as f:
        data = json.load(f)
    
    none_path =[]
    for anno in tqdm.tqdm(data):
        video_name = anno['video_name']
        
        if os.path.exists(os.path.join(file_dir, "v_{}.mp4".format(video_name))):
            continue
        else:
            if os.path.exists(os.path.join(file_dir, "v_{}.mkv".format(video_name))):
                continue
            else:
                if os.path.exists(os.path.join(file_dir, "v_{}.webm".format(video_name))):
                    continue
                else:
                    none_path.append(os.path.join(file_dir, "v_{}.mp4".format(video_name)))
                    print("v_{}.mp4 or mkv".format(video_name))
                    print("path not exist !!")
    print(none_path)
    print(len(none_path))