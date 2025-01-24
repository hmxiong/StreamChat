import cv2
import os

def check_video_file(path):
    # 检查文件是否存在
    if not os.path.exists(path):
        return f"文件 {path} 不存在。"
    
    # 尝试用 OpenCV 打开视频文件
    video = cv2.VideoCapture(path)
    if not video.isOpened():
        return f"无法打开视频文件: {path}"
    
    # 关闭视频文件
    video.release()
    return f"视频文件 {path} 可以正常打开。"

# 示例使用
video_path = '/13390024681/All_Data/ActNet-Video/all_test/v_4WGjeXTgpis.mkv'
result = check_video_file(video_path)
print(result)
