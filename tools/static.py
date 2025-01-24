import json
import os
import cv2
import tqdm

from collections import defaultdict
import matplotlib.pyplot as plt
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, \
                         LlamaConfig, LlamaModel, LlamaForCausalLM, pipeline

def time_distribution():
    llama_path = "/13390024681/All_Model_Zoo/llama3-8b-instruct-hf"
    llama_tokenizer = AutoTokenizer.from_pretrained(llama_path, use_fast=True)
    # 假设你的 JSON 文件名为 data.json
    json_file_path = '/13390024681/llama/EfficientVideo/Ours/streaming_bench_v0.3.json'

    video_path_dict = {
            "/13390024681/All_Data/Streaming_final":["Cooking_show" ,"Comedy_(drama)" ,"Apple_TV"], 
            "/13390024681/All_Data/Supplement_1":["Cooking", "Metalworking"]
            }

    def find_key_by_category(category, data_dict):
        for key, categories in data_dict.items():
            if category in categories:
                return key
        return None
        
    # 读取 JSON 文件
    with open(json_file_path, 'r') as file:
        data = json.load(file)

    # 初始化统计变量
    total_questions = 0
    class_count = defaultdict(int)
    total_video_time = 0
    time_distribution = []
    total_question_tokens = 0
    total_answer_tokens = 0
    num_questions = 0
    num_answers = 0

    # 遍历每个元素，统计问题数量及类别，视频时长及分布，问题和回答的 token 数量
    for item in data:
        breakpoints = item.get('breakpoint', [])
        total_questions += len(breakpoints)
        
        # 获取视频路径并计算视频时长
        video_path = item['info']['video_path']
        class_1 = item['info']['class_1']
        class_2 = item['info']['class_2']

        file_dir = "/13390024681/All_Data/Streaming_Bench_v0.3"
        file_dir = os.path.join(file_dir, class_1)
        
        video_path = os.path.join(file_dir, video_path)  
        # print(video_path)  
        if os.path.exists(video_path):
            cap = cv2.VideoCapture(video_path)
            if cap.isOpened():
                fps = cap.get(cv2.CAP_PROP_FPS)
                frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
                video_duration_seconds = frame_count / fps
                video_duration_minutes = video_duration_seconds / 60  # 转化为分钟
                total_video_time += video_duration_minutes
                time_distribution.append(video_duration_minutes)
            cap.release()
        else:
            print("file not exist:{}".format(video_path))
        
        for bp in breakpoints:
            class_type = bp.get('class')
            if class_type:
                class_count[class_type] += 1
            
            # 统计问题和回答的 token 数量
            question_tokens = llama_tokenizer.encode(bp.get('question', ''), add_special_tokens=False)
            answer_tokens = llama_tokenizer.encode(bp.get('answer', ''), add_special_tokens=False)
            total_question_tokens += len(question_tokens)
            total_answer_tokens += len(answer_tokens)
            num_questions += 1
            num_answers += 1

    # 输出统计结果
    print(f"总共的breakpoint问题数量: {total_questions}")
    print("每个类别的问题数量:")
    for class_type, count in class_count.items():
        print(f"类别 {class_type}: {count} 问题")

    print(f"所有视频的总时长: {total_video_time} mine")
    print(f"时长分布: {time_distribution}")
    print(f"问题的总token数量: {total_question_tokens}")
    print(f"回答的总token数量: {total_answer_tokens}")
    print(f"问题的平均token数量: {total_question_tokens / num_questions}")
    print(f"回答的平均token数量: {total_answer_tokens / num_answers}")

    # 统计不同时间区间的视频数量
    time_bins = [0, 1, 2, 3, 4, 5, 6, 7]  # 定义时间区间（分钟）
    time_bin_labels = ["0-5", "5-10", "10-15", "15-20", "20-30", "30-60", "60-120"]
    time_bin_counts = defaultdict(int)

    for time in time_distribution:
        for i in range(len(time_bins) - 1):
            if time_bins[i] <= time < time_bins[i + 1]:
                time_bin_counts[time_bin_labels[i]] += 1
                break

    # 输出时间区间统计
    print("视频时长在不同时间区间内的数量:")
    for label, count in time_bin_counts.items():
        print(f"时长区间 {label} 分钟: {count} 个视频")
        
    # 准备数据绘制饼图
    labels = list(class_count.keys())
    sizes = [count / total_questions * 100 for count in class_count.values()]


    fig, ax = plt.subplots(figsize=(10, 10), dpi=100)
    ax.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
    ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

    # 设置标题
    plt.title('各类别问题数量占总数的比例')

    # 保存饼图
    plt.savefig('class_distribution_pie_chart.png')

    # 绘制时长分布直方图
    fig, ax = plt.subplots(figsize=(12, 8), dpi=100)
    ax.hist(time_distribution, bins=20, edgecolor='black')
    # plt.hist(time_distribution, bins=20, edgecolor='black')
    plt.title('Video Time Distribute')
    plt.xlabel('Time(mine)')
    plt.ylabel('count')
    plt.savefig('video_time_distribution_histogram.png')
    # plt.show()

def text_length_distribution():
    llama_path = "/13390024681/All_Model_Zoo/llama3-8b-instruct-hf"
    llama_tokenizer = AutoTokenizer.from_pretrained(llama_path, use_fast=True)
    # 假设你的 JSON 文件名为 data.json
    json_file_path = '/13390024681/llama/EfficientVideo/Ours/streaming_bench_v0.3.json'

    video_path_dict = {
            "/13390024681/All_Data/Streaming_final":["Cooking_show" ,"Comedy_(drama)" ,"Apple_TV"], 
            "/13390024681/All_Data/Supplement_1":["Cooking", "Metalworking"]
            }

    def find_key_by_category(category, data_dict):
        for key, categories in data_dict.items():
            if category in categories:
                return key
        return None
        
    # 读取 JSON 文件
    with open(json_file_path, 'r') as file:
        data = json.load(file)

    # 初始化统计变量
    total_questions = 0
    class_count = defaultdict(int)
    total_video_time = 0
    time_distribution = []
    total_question_tokens = 0
    total_answer_tokens = 0
    num_questions = 0
    num_answers = 0
    
    # 用于存储每组问题和回答的token数量
    question_token_lengths = []
    answer_token_lengths = []

    # 遍历每个元素，统计问题数量及类别，视频时长及分布，问题和回答的 token 数量
    for item in tqdm.tqdm(data):
        breakpoints = item.get('breakpoint', [])
        total_questions += len(breakpoints)
        
        # 获取视频路径并计算视频时长
        video_path = item['info']['video_path']
        class_1 = item['info']['class_1']
        class_2 = item['info']['class_2']

        file_dir = "/13390024681/All_Data/Streaming_Bench_v0.3"
        file_dir = os.path.join(file_dir, class_1)
    
        
        for bp in breakpoints:
            class_type = bp.get('class')
            if class_type:
                class_count[class_type] += 1
            
            # 统计问题和回答的 token 数量
            question_tokens = llama_tokenizer.encode(bp.get('question', ''), add_special_tokens=False)
            answer_tokens = llama_tokenizer.encode(bp.get('answer', ''), add_special_tokens=False)
            
            question_token_lengths.append(len(question_tokens))
            answer_token_lengths.append(len(answer_tokens))
            
            total_question_tokens += len(question_tokens)
            total_answer_tokens += len(answer_tokens)
            num_questions += 1
            num_answers += 1

    # 输出统计结果
    print(f"总共的breakpoint问题数量: {total_questions}")
    print("每个类别的问题数量:")
    for class_type, count in class_count.items():
        print(f"类别 {class_type}: {count} 问题")

    print(f"问题的总token数量: {total_question_tokens}")
    print(f"回答的总token数量: {total_answer_tokens}")
    print(f"问题的平均token数量: {total_question_tokens / num_questions}")
    print(f"回答的平均token数量: {total_answer_tokens / num_answers}")
    
    # 绘制直方图
    plt.figure(figsize=(12, 6))
    
    # 绘制问题token数量分布直方图
    plt.subplot(1, 2, 1)
    plt.hist(question_token_lengths, bins=20, color='blue', alpha=0.7)
    plt.title('Question Token Length Distribution')
    plt.xlabel('Token Length')
    plt.ylabel('Frequency')
    
    # 保存问题token数量分布直方图
    plt.savefig('question_token_length_distribution.png')
    plt.close()  # 关闭当前图像，防止后续操作影响
    
    # 绘制回答token数量分布直方图
    plt.subplot(1, 2, 2)
    plt.hist(answer_token_lengths, bins=20, color='green', alpha=0.7)
    plt.title('Answer Token Length Distribution')
    plt.xlabel('Token Length')
    plt.ylabel('Frequency')
    
    # 保存回答token数量分布直方图
    plt.savefig('answer_token_length_distribution.png')
    plt.close()  # 关闭当前图像
    
    # 展示图像
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    text_length_distribution()