import os
import json
import tqdm

if __name__ == '__main__':
    
    import os
    from datasets import load_dataset
    from tqdm import tqdm
    import json

    data = load_dataset("/13390024681/All_Data/ActiveNet-json", split="test")
    # data = load_dataset("/15324359926/WeightMerge/llava-next-data", split="train")

    image_folder = "/15324359926/datasets/llava_data"
    # image_folder = "/15324359926/WeightMerge/llava-next-data/images"

    converted_data = []
    number = 1
    for da in tqdm(data):
        print(da)
        assert 1==2
        # print(da)
        # formatted_id = f"vqa_rad_{number:09}"
        # # print(formatted_id)
        # json_data = {}
        # json_data["id"] = formatted_id
        # if da["image"] is not None:
        #     json_data["image"] = f"vqa_rad/images/{formatted_id}.jpg"
        #     if not os.path.exists(os.path.join(image_folder, json_data["image"])):
        #         parent_directory = "/".join(os.path.join(image_folder, json_data["image"]).split("/")[:-1])
        #         # print('parent_directory',parent_directory)
        #         if not os.path.exists(parent_directory):
        #             os.makedirs(parent_directory)
        #         da["image"].convert('RGB').save(os.path.join(image_folder, json_data["image"]))  # json_data["image"]  = vqa_rad/images/{formatted_id}.jpg


        # # 改conversation
        # question = da['question']
        # question = question[0].upper() + question[1:]
        # answer = da['answer']
        # json_data["conversations"] = [{'from': 'human', 'value': f'<image>\n{question}\nAnswer the question using a single word or phrase.'}, {'from': 'gpt', 'value': f'{answer}'}]
        # converted_data.append(json_data)
        # number = number + 1
        # # print(json_data)
        # # assert 1 ==2
        # # break

    # with open("/15324359926/datasets/llava_data/vqa_rad/vqa_rad_instruction_train.json", "w") as f:
    #     json.dump(converted_data, f, indent=4, ensure_ascii=False)
    