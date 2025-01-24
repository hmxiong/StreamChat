import os
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt 
from PIL import Image
from llavanext.mm_utils import tokenizer_image_token, get_model_name_from_path
from torch_kmeans import KMeans
from memory_bank.memory_utils import summarize_memory_event_personality, enter_name, save_local_memory, HfArgumentParser, DataArguments, ModelArguments
from memory_bank.prompt_utils import *
from memory_bank.summarize_memory import LLMClientLLaMA3
from longva.conversation import conv_templates, SeparatorStyle
from transformers import EvalPrediction, Trainer
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, AutoModel
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from sentence_transformers.util import cos_sim
from collections import deque, defaultdict
# from sentence_transformers import SentenceTransformer
# from sentence_transformers.util import cos_sim
# from sentence_transformers.quantization import quantize_embeddings
import random
import math
import time
import numpy as np
import re
import json
import tqdm
import threading

# 定义颜色代码
RED = "\033[31m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
BLUE = "\033[34m"
PURPLE = "\033[35m"
CYAN = "\033[36m"
WHITE = "\033[37m"
RESET = "\033[0m"

class TreeNode:
    def __init__(self, centroids, labels=None, depth=0):
        self.centroids = centroids
        self.labels = labels
        self.children = []
        self.depth = depth
        
class MultimodalTreeNode:
    def __init__(self, centroids, text, text_distance=None, image_distance=None, labels=None, depth=0):
        self.centroids = centroids
        self.text = text
        self.text_distance = text_distance
        self.image_distance = image_distance
        self.labels = labels
        self.children = []
        self.depth = depth

def expand2square(pil_img, background_color):
    width, height = pil_img.size
    if width == height:
        return pil_img
    elif width > height:
        result = Image.new(pil_img.mode, (width, width), background_color)
        result.paste(pil_img, (0, (width - height) // 2))
        return result
    else:
        result = Image.new(pil_img.mode, (height, height), background_color)
        result.paste(pil_img, ((height - width) // 2, 0))
        return result
    
def process_images(images, image_processor, model_cfg):
    # image_aspect_ratio = getattr(model_cfg, "image_aspect_ratio", None)
    new_images = []
    
    for image in images:
        # image = expand2square(image, tuple(int(x * 255) for x in image_processor.image_mean))
        image = image_processor.preprocess(image, return_tensors="pt")["pixel_values"][0]
        new_images.append(image) # [ dim resize_w resize_h ]
    if len(images) > 1:
        new_images = [torch.stack(new_images, dim=0)] # [num_images dim resize_w resize_h ]
    
    # print("len new_images:{}".format(len(new_images)))
    if all(x.shape == new_images[0].shape for x in new_images):
        new_images = torch.stack(new_images, dim=0) # num_image num_patches dim resize_w resize_h
        # when using "pad" mode and only have one image the new_images tensor dimension is [ 1 dim resize_w resize_h ]
    # print("new_images:{}".format(new_images.shape))
    return new_images

def compute_gradients(img):
    sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32, device=img.device).unsqueeze(0).unsqueeze(0)
    sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32, device=img.device).unsqueeze(0).unsqueeze(0)
    
    Ix = F.conv2d(img.unsqueeze(0), sobel_x, padding=1)
    Iy = F.conv2d(img.unsqueeze(0), sobel_y, padding=1)
    
    return Ix.squeeze(0), Iy.squeeze(0)

def Optical_flow(last_frame, current_frame, image_processor, model, threshold, device='cuda:0'):
    window_size = 5
    # threshold = 0.21  # 设置光流法阈值
    eps = 1e-6
    
    assert last_frame is not None
    
    current_image_tensor = process_images([Image.fromarray(current_frame)], image_processor, model.config).to(device).squeeze(0)
    # [1 3 h w]
    last_image_tensor = process_images([Image.fromarray(last_frame)], image_processor, model.config).to(device).squeeze(0)
    
    if current_image_tensor.dim() == 3:
        current_image_tensor_gray = 0.2989 * current_image_tensor[0, :, :] + 0.5870 * current_image_tensor[1, :, :] + 0.1140 * current_image_tensor[2, :, :]
        last_image_tensor_gray = 0.2989 * last_image_tensor[0, :, :] + 0.5870 * last_image_tensor[1, :, :] + 0.1140 * last_image_tensor[2, :, :]
    else:
        current_image_tensor_gray = current_image_tensor
        last_image_tensor_gray = last_image_tensor
    
    # Compute gradients on GPU
    Ix, Iy = compute_gradients(last_image_tensor_gray.unsqueeze(0))
    It = current_image_tensor_gray - last_image_tensor_gray
    # print("Ix:{}, Iy:{}, It:{}".format(Ix.shape, Iy.shape, It.shape))
    # Initialize flow vectors on GPU
    u = torch.zeros_like(Ix, device=Ix.device)
    v = torch.zeros_like(Ix, device=Ix.device)
    
    w = window_size // 2
    
    # Prepare for batch processing
    Ix_windows = F.unfold(Ix.unsqueeze(0), kernel_size=(window_size, window_size)).transpose(1, 2)
    Iy_windows = F.unfold(Iy.unsqueeze(0), kernel_size=(window_size, window_size)).transpose(1, 2)
    It_windows = F.unfold(It.unsqueeze(0).unsqueeze(0), kernel_size=(window_size, window_size)).transpose(1, 2)
    
    A = torch.stack((Ix_windows, Iy_windows), dim=3)
    b = -It_windows
    
    # # Solve for flow vectors in batch
    # nu = torch.linalg.solve(A_T_A, A_T_b) # 出现非奇异解，导致无法计算出结果
    
    # Using Lucas-Karthy method
    # Reshape to (batch_size, num_windows, window_size*window_size, 2)
    A = A.view(A.size(0), -1, window_size*window_size, 2)
    b = b.view(b.size(0), -1, window_size*window_size)
    
    # Compute A^T * A and A^T * b
    A_T_A = torch.matmul(A.transpose(2, 3), A)
    A_T_b = torch.matmul(A.transpose(2, 3), b.unsqueeze(3)).squeeze(3)
    
    # Add regularization term to A_T_A
    eye = torch.eye(A_T_A.size(-1), device=A_T_A.device)
    A_T_A += eps * eye
    
    # Solve for flow vectors in batch
    nu = torch.linalg.solve(A_T_A, A_T_b)
    
    u_flat = nu[:, :, 0]
    v_flat = nu[:, :, 1]

    # Reshape flow vectors to image shape
    # Calculate correct output size for fold
    output_height = Ix.shape[1] - window_size + 1
    output_width = Ix.shape[2] - window_size + 1
    
    # Ensure the data is suitable for fold operation
    u_flat = u_flat.view(1, output_height,  output_width)
    v_flat = v_flat.view(1, output_height,  output_width)
    
    # Compute magnitude of flow vectors
    mag = torch.sqrt(u_flat**2 + u_flat**2)
    mean_mag = mag.mean().item()

    del Ix_windows
    del Iy_windows
    del It_windows

    del current_image_tensor_gray
    del last_image_tensor_gray
    del last_image_tensor
    
    if mean_mag > threshold:
        return True, mean_mag, current_image_tensor
    else:
        return False, mean_mag, current_image_tensor

def SSIM(last_frame, current_frame, image_processor, model, threshold, window_size=11, sigma=1.5, device='cuda'):
    # 将图像转换为张量，并将其发送到指定的设备上
    # current_image_tensor_gray = img1.to(device)
    # last_image_tensor_gray = img2.to(device)
    assert last_frame is not None
    
    current_image_tensor = process_images([Image.fromarray(current_frame)], image_processor, model.config).cuda().squeeze(0)
    # [1 3 h w]
    last_image_tensor = process_images([Image.fromarray(last_frame)], image_processor, model.config).cuda().squeeze(0)
    
    if current_image_tensor.dim() == 3:
        current_image_tensor_gray = 0.2989 * current_image_tensor[0, :, :] + 0.5870 * current_image_tensor[1, :, :] + 0.1140 * current_image_tensor[2, :, :]
        last_image_tensor_gray = 0.2989 * last_image_tensor[0, :, :] + 0.5870 * last_image_tensor[1, :, :] + 0.1140 * last_image_tensor[2, :, :]
    else:
        current_image_tensor_gray = current_image_tensor
        last_image_tensor_gray = last_image_tensor
    # print("current_image_tensor_gray:{}".format(current_image_tensor_gray.shape))
    current_image_tensor_gray = current_image_tensor_gray.unsqueeze(0).unsqueeze(0)
    last_image_tensor_gray = last_image_tensor_gray.unsqueeze(0).unsqueeze(0)
    # # 函数用于计算局部的高斯权重
    # def gaussian(window_size, sigma):
    #     # gauss = torch.Tensor([torch.exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
    #     gauss = torch.Tensor([torch.exp(-(x - window_size//2)**2 / (2*sigma**2)) for x in range(window_size)])
    #     return gauss/gauss.sum()
    def gaussian(window_size, sigma):
        gauss = torch.Tensor([torch.exp(torch.tensor(-(x - window_size//2)**2 / (2*sigma**2))) for x in range(window_size)])
        return gauss / gauss.sum()
    
    # 计算 SSIM 的局部高斯权重
    # window = gaussian(window_size, sigma).unsqueeze(1).unsqueeze(2).to(device)
    # 计算高斯核
    _1D_window = gaussian(window_size, sigma).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = _2D_window.to(device)

    # print("window:{}".format(window.shape))
    # print("current_image_tensor_gray:{}".format(current_image_tensor_gray.shape))
    # assert 1==2
    # 计算均值
    mu1 = F.conv2d(input=current_image_tensor_gray, weight=window, stride=1, padding=window_size//2)
    mu2 = F.conv2d(input=last_image_tensor_gray,  weight=window, stride=1, padding=window_size//2)
    
    # 计算方差
    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

     # 计算图像的标准差
    sigma1_sq = F.conv2d(input=current_image_tensor_gray * current_image_tensor_gray, weight=window, stride=1, padding=window_size//2) - mu1_sq
    sigma2_sq = F.conv2d(input=last_image_tensor_gray * last_image_tensor_gray, weight=window, stride=1, padding=window_size//2) - mu2_sq
    sigma12 = F.conv2d(input=current_image_tensor_gray * last_image_tensor_gray, weight=window, stride=1, padding=window_size//2) - mu1_mu2

    # 常数 C1 和 C2
    C1 = (0.01)**2
    C2 = (0.03)**2

    # 计算 SSIM
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    # 对 SSIM 进行平均池化，得到最终的 SSIM 分数
    ssim_score = ssim_map.mean()

    # return ssim_score
    if ssim_score > threshold:
        return True, ssim_score, current_image_tensor
    else:
        return False, ssim_score, current_image_tensor


def calculate_forgetting_probabilities(length, tau=10):
    """根据艾宾浩斯遗忘曲线计算每个位置的遗忘概率"""
    t = np.arange(length)
    R_t = np.exp(-t / tau)
    return R_t / R_t.sum()

def select_data_without_replacement(queue, probabilities, selection_length=10):
    """根据遗忘概率从队列中选择数据，确保没有重复"""
    indices = np.arange(len(queue))
    selected_indices = np.random.choice(indices, size=selection_length, replace=False, p=probabilities)
    selected_data = [queue[idx] for idx in selected_indices]
    return selected_data

def compress_spatial_features(feature_list, compress_rate):
    
    comperssed_spatial_features = []
    
    assert len(feature_list) > 0, "compressed feature must larger then 0"
    
    all_features = torch.cat(feature_list)
    patch_size = round(math.sqrt(all_features.shape[1]))
    # print(fea)
    B, SEQ, DIM = all_features.shape
    # print("all feature:{}".format(all_features.shape))
    
    assert patch_size * patch_size == feature_list[0].shape[1], f"For ViT feature map, {patch_size}*{patch_size}={patch_size**2} != {all_features.shape[1]}"
    all_features = all_features.reshape(-1, patch_size, patch_size, DIM).permute(0, 3, 1, 2)
    # print("all feature:{}".format(all_features.shape))
    pooled_features = F.avg_pool2d(all_features, (compress_rate, compress_rate)).permute(0, 2, 3, 1)
    pooled_features = pooled_features.reshape(B, -1, DIM)
    
    # print("pooled features:{}".format(pooled_features.shape))
    # for feature in feature_list:
    #     feature = feature.reshape(-1, patch_size, patch_size, DIM).permute(0, 3, 1, 2)
    #     feature = F.avg_pool2d(feature, (patch_size // compress_rate, patch_size // compress_rate))
    #     feature = feature.permute(0, 2, 3, 1)
    #     feature = feature.reshape(B, -1, DIM) # 1 comperss*comperss DIM
    comperssed_spatial_features = list(torch.split(pooled_features, 1))
    return comperssed_spatial_features

def weighted_kmeans_feature(img_feature, video_max_frames, weights=None):
    if weights is None:
        weights = torch.ones(img_feature.size(0), dtype=img_feature.dtype, device=img_feature.device)
    def weighted_kmeans_torch(X, num_clusters, weights=None, distance='euclidean', tol=1e-4, max_iter=10):
        indices = torch.randperm(X.size(0), device=X.device)[:num_clusters]
        centroids = X[indices]
        for i in range(max_iter):
            if distance == 'euclidean':
                dists = ((X.unsqueeze(1) - centroids.unsqueeze(0)) ** 2).sum(dim=2).sqrt()
            else:
                raise NotImplementedError("Only Euclidean distance is supported yet")
            labels = torch.argmin(dists, dim=1)
            weighted_sum = torch.zeros_like(centroids)
            weights_sum = torch.zeros(num_clusters, dtype=X.dtype, device=X.device)
            for j in range(num_clusters):
                cluster_mask = labels == j
                weighted_sum[j] = torch.sum(weights[cluster_mask, None] * X[cluster_mask], dim=0)
                weights_sum[j] = torch.sum(weights[cluster_mask])
            mask = weights_sum > 0
            new_centroids = torch.zeros_like(weighted_sum)
            new_centroids[mask] = weighted_sum[mask] / weights_sum[mask, None]
            if mask.sum() < num_clusters:  # fix nan centroids
                new_centroids[~mask] = torch.stack([X[random.randint(0, X.size(0) - 1)] for _ in range(num_clusters - mask.sum())])
            diff = torch.norm(centroids - new_centroids, dim=1).sum()
            if diff < tol:
                break
            centroids = new_centroids
        return centroids, labels, weights_sum, i
    T, P, D = img_feature.shape
    T0 = video_max_frames
    if T <= T0:
        return img_feature, weights, [[[i] for i in range(T)]]
    X = img_feature.view(T, -1)  # [T, P, D]
    centroids, labels, weights, exit_step = weighted_kmeans_torch(X, T0, weights)
    reduced_feature = centroids.view(T0, P, D)
    # print(f'Note: perform weighted kmeans feature {img_feature.shape} to {reduced_feature.shape}, exit at step={exit_step}')  # actually, K=T0
    step_indices = [[] for _ in range(T0)]
    for i in range(T0):
        step_indices[i] = [j for j in range(T) if labels[j] == i]
    return reduced_feature, labels

def k_means_clustering(X, num_clusters, max_iter=10):
    """对输入的张量进行 K-means 聚类"""
    indices = torch.randperm(X.size(0), device=X.device)[:num_clusters]
    centroids = X[indices]
    for i in range(max_iter):
        dists = ((X.unsqueeze(1) - centroids.unsqueeze(0)) ** 2).sum(dim=2).sqrt()
        labels = torch.argmin(dists, dim=1)
        new_centroids = torch.stack([X[labels == j].mean(dim=0) for j in range(num_clusters)])
        if torch.allclose(new_centroids, centroids):
            break
        centroids = new_centroids
    return centroids, labels


def building_memory_tree(feature_list, compress_rate, chunk_size, num_clusters, depth):
    
    time_1 = time.time()
    
    feature_list = compress_spatial_features(feature_list, compress_rate)
    
    time_2 = time.time()
    
    chunk_feature_list = [feature_list[i:i + chunk_size] for i in range(0, len(feature_list), chunk_size)]
    
    time_3 = time.time()
    # 对每个子序列进行 K-means 聚类
    nodes = []
    for sub_seq in chunk_feature_list:
        sub_seq_feature = torch.cat(sub_seq, dim=0)
        # print(sub_seq_feature.shape) # 
        [reduced_feature, weights, step_indices, labels] = weighted_kmeans_feature(sub_seq_feature, num_clusters)
        node = TreeNode(reduced_feature, labels, depth)
        nodes.append(node)
        
    time_4 = time.time() # 主要时间花销发生在这个一步
    
    # 对每个节点的聚类中心递归地构建树
    for node in nodes:
        if node.centroids.size(0) > num_clusters:
            child_node = building_memory_tree(node.centroids, compress_rate, chunk_size, num_clusters, depth + 1)
            node.children.append(child_node)
    
    time_5 = time.time()
    
    print("time spend:{}/{}/{}/{}".format((time_2-time_1), (time_3-time_2), (time_4-time_3), (time_5-time_4)))
    
    return nodes

def buildingd_memory_tree_buttom_up(k_means_chunk_feature_list, num_clusters, interval):
    
    """从底层到顶层构建树状结构"""
    nodes = [TreeNode(tensor, depth=0) for tensor in k_means_chunk_feature_list]
    
    while len(nodes) > 1:
        new_nodes = []
        for i in range(0, len(nodes), interval):
            chunk = nodes[i:i + interval]
            centroids_list = [node.centroids for node in chunk] # len 10 144 1024
            combined_centroids = torch.cat(centroids_list, dim=0) # 
            if combined_centroids.shape[0] > num_clusters:
                new_centroids, labels = weighted_kmeans_feature(combined_centroids, num_clusters)
            else:
                new_centroids = combined_centroids
            new_node = TreeNode(new_centroids, depth=chunk[0].depth + 1)
            for j, node in enumerate(chunk):
                new_node.children.append(node)
            new_nodes.append(new_node)
        nodes = new_nodes
    
    return nodes[0]

def buildingd_memory_tree_buttom_up_with_summarize_token(k_means_chunk_feature_list, num_clusters, interval, summarizer, input_ids, tokenizer, chunked_feature_list):
    
    def make_summary_prompt(caption_list):
        order = ["first", "second", "third", "fourth", "fifth", "sixth", "seventh", "eighth", "ninth", "tenth"]
        
        new_caption =[]
        
        for index, caption in enumerate(caption_list):
            new_caption.append("The caption of the {} video clip is:{} \n".format(order[index], caption))
        
        qs = " ".join(new_caption)
        qs = "You need to write a summary of the following, including as many key details as possible into one sentence." + qs
        conv = conv_templates["qwen_1_5_summarize"].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        summarize_prompt = conv.get_prompt()
        # print(question)
        # captioning_input_ids = tokenizer_image_token(captioning_prompt, summarizer_tokenzier, IMAGE_TOKEN_INDEX, return_tensors='pt')
        summarize_ids = torch.tensor(tokenizer(summarize_prompt).input_ids , dtype=torch.long)
        summarize_ids = summarize_ids.unsqueeze(0).cuda()
        return summarize_ids
    
    """从底层到顶层构建树状结构"""
    # base_captions = [summarizer.]
    output_list = [] # prepare summary
    for chunk_feature in chunked_feature_list: # len 
        dimension = chunk_feature[0].shape[-1]
        chunk_feature = torch.cat(chunk_feature, dim=0).reshape(-1, dimension)
        with torch.no_grad():
            output_ids = summarizer.generate_with_image_embedding(
                input_ids,
                image_embeddings=[chunk_feature],
                modalities=["video"],
                # question_ids=ques_ids,
                # modalities="image",
                do_sample=True ,
                temperature=0.1,
                top_p=None,
                num_beams=1,
                max_new_tokens=128,
                use_cache=False)

        outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
        output_list.append(outputs)
        
    nodes = [MultimodalTreeNode(tensor, text, depth=0) for (tensor, text) in zip(k_means_chunk_feature_list, output_list)]
    
    while len(nodes) > 1:
        new_nodes = []
        for i in range(0, len(nodes), interval):
            chunk = nodes[i:i + interval]
            centroids_list = [node.centroids for node in chunk] # len 10 144 1024
            caption_list = output_list[i:i+interval]
            combined_centroids = torch.cat(centroids_list, dim=0) # 
            
            if combined_centroids.shape[0] > num_clusters:
                new_centroids, labels = weighted_kmeans_feature(combined_centroids, num_clusters)
            else:
                new_centroids = combined_centroids
            summarize_ids = make_summary_prompt(caption_list)
            # making summarize for tree model
            with torch.no_grad():
                output_ids = summarizer.generate_with_image_embedding(
                    summarize_ids,
                    image_embeddings= None,
                    modalities=["video"],
                    # question_ids=ques_ids,
                    # modalities="image",
                    do_sample=True ,
                    temperature=0.1,
                    top_p=None,
                    num_beams=1,
                    max_new_tokens=256,
                    use_cache=False)

            summarize_text = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
            new_node = MultimodalTreeNode(new_centroids, summarize_text, depth=chunk[0].depth + 1)
            
            for j, node in enumerate(chunk):
                new_node.children.append(node)
            new_nodes.append(new_node)
        nodes = new_nodes
    
    return nodes[0]


def fast_building_memory_tree_summarize_token(k_means_chunk_feature_list, num_clusters, interval, summarizer, input_ids, tokenizer, chunked_feature_list, existing_tree=None):
    
    def print_tree_x(node, indent=0):
        """打印树状结构"""
        if isinstance(node, list):
            for single_node in node:
                print(" " * indent + f"{RED}Depth{RESET}: {single_node.depth}, {RED}Centroids{RESET}: {single_node.centroids.shape}, {RED}Text{RESET} :{single_node.text}")
                if single_node.children is not None:
                    for child in single_node.children:
                        print_tree_x(child, indent + 4)
                
        else:
            print(" " * indent + f"{RED}Depth{RESET}: {node.depth}, {RED}Centroids{RESET}: {node.centroids.shape}, {RED}Text{RESET} :{node.text}")
            for child in node.children:
                print_tree_x(child, indent + 4)
                
    def make_summary_prompt(caption_list):
        order = ["first", "second", "third", "fourth", "fifth", "sixth", "seventh", "eighth", "ninth", "tenth"]
        
        new_caption =[]
        
        for index, caption in enumerate(caption_list):
            new_caption.append("The caption of the {} video clip is:{} \n".format(order[index], caption))
        
        qs = " ".join(new_caption)
        qs = "You need to write a summary of the following, including as many key details as possible into one sentence." + qs
        conv = conv_templates["qwen_1_5_summarize"].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        summarize_prompt = conv.get_prompt()
        # print(question)
        # captioning_input_ids = tokenizer_image_token(captioning_prompt, summarizer_tokenzier, IMAGE_TOKEN_INDEX, return_tensors='pt')
        summarize_ids = torch.tensor(tokenizer(summarize_prompt).input_ids , dtype=torch.long)
        summarize_ids = summarize_ids.unsqueeze(0)
        return summarize_ids
    
    def get_summarize_depth(nodes, interval):
        depth_count = defaultdict(int)
        depth_count.clear()
        for node in nodes:
            depth_count[node.depth] += 1
                
        # 找出最高优先级的深度
        max_depth = max(depth_count.keys())
        for depth in range(max_depth, -1, -1):
            if depth_count[depth] % interval == 0 and depth_count[depth] > 0: # 判断目前需要使用的深度
                return depth, depth_count
        return 0 ,depth_count
    
    output_list = []  # prepare summary
    for chunk_feature in chunked_feature_list:
        dimension = chunk_feature[0].shape[-1]
        chunk_feature = torch.cat(chunk_feature, dim=0).reshape(-1, dimension).to(summarizer.device)
        # print("chunk_feature", chunk_feature.shape)
        # time.sleep(2)  # 模拟计算时间
        with torch.no_grad():
            output_ids = summarizer.generate_with_image_embedding(
                input_ids.to(summarizer.device),
                image_embeddings=[chunk_feature],
                modalities=["video"],
                # question_ids=ques_ids,
                # modalities="image",
                do_sample=True ,
                temperature=0.1,
                top_p=None,
                # num_beams=1,
                max_new_tokens=128,
                use_cache=False)

        outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
        output_list.append(outputs)
    
    nodes = [MultimodalTreeNode(tensor, text, depth=0) for (tensor, text) in zip(k_means_chunk_feature_list, output_list)]
    
    if existing_tree:
        # nodes.append(existing_tree)
        nodes = existing_tree + nodes
    
    summarize_depth, depth_count = get_summarize_depth(nodes, interval)
    
    start_index = next((index for index, node in enumerate(nodes) if node.depth == summarize_depth), None)
    chunk_length = len([x for x in nodes if x.depth == summarize_depth])
    print("summarize_depth:{}/ start_index:{}/ chunk_length:{} / len(nodes):{}".format(summarize_depth, start_index, chunk_length, len(nodes)))
    
    # if chunk_length % interval >= 0 and len(nodes) > 0: # for first
    if chunk_length % interval >= 0 and len(nodes) > 0 and chunk_length >= interval:
                    
        print("building summarize node for {} clip".format(start_index + 1))
        
        
        # for i in range(0 + clip * interval, len(nodes), interval):
        # chunk = nodes[(clip - 1) * interval : (clip - 1) * interval + interval]
        chunk = nodes[start_index: start_index + interval]
        centroids_list = [node.centroids for node in chunk]
        caption_list = [node.text for node in chunk]
        combined_centroids = torch.cat(centroids_list, dim=0)

        if combined_centroids.shape[0] > num_clusters:
            new_centroids, labels = weighted_kmeans_feature(combined_centroids, num_clusters)
        else:
            new_centroids = combined_centroids

        summarize_ids = make_summary_prompt(caption_list)
        
        with torch.no_grad():
            output_ids = summarizer.generate_with_image_embedding(
                summarize_ids.to(summarizer.device),
                image_embeddings= None,
                modalities=["video"],
                # question_ids=ques_ids,
                # modalities="image",
                do_sample=True ,
                temperature=0.1,
                top_p=None,
                # num_beams=1,
                max_new_tokens=256,
                use_cache=False)

        summarize_text = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
        # time.sleep(2)  # 模拟计算时间
        new_node = MultimodalTreeNode(new_centroids, summarize_text, depth=chunk[0].depth + 1)

        for j, node in enumerate(chunk):
            new_node.children.append(node) # 将之前的list全部进行总结
        
        nodes[start_index: start_index + interval] = [new_node]
        print_tree_x(nodes)
        return nodes
    
    else:
        print_tree_x(nodes)
        return nodes

def search_tree_multi_modal_with_embedding(node, query, image_embedding, model, tokenizer, top_k=1):
    """在树中沿着每一层相似度最高的分支不断寻找，并将每一层对应的特征都提取出来"""
    # The model works really well with cls pooling (default) but also with mean pooling.
    def pooling(outputs: torch.Tensor, inputs,  strategy: str = 'cls') -> np.ndarray:
        if strategy == 'cls':
            outputs = outputs[:, 0]
        elif strategy == 'mean':
            outputs = torch.sum(
                outputs * inputs["attention_mask"][:, :, None], dim=1) / torch.sum(inputs["attention_mask"])
        else:
            raise NotImplementedError
        return outputs.detach().cpu().numpy()
    
    path_features = []
    path_text = []
    # x = torch.nn.functional.normalize(torch.cat([query, image_embedding], dim=0), p=2, dim=0)
    print("Question:{}".format(query))
    query_ids = tokenizer(query, padding=True, return_tensors='pt')
    for k, v in query_ids.items():
        query_ids[k] = v.cuda()
    outputs_1 = model(**query_ids).last_hidden_state
    query_embedding = pooling(outputs_1, query_ids, 'cls')
    
    current_node = node
    while current_node.children:
        best_child_index = None
        best_sim = 0
        
        # 计算查询张量与每个子节点的所有特征张量之间的相似度
        for i, child in enumerate(current_node.children):
            # distances = torch.cdist(query.unsqueeze(0), child.centroids.view(-1, query.size(-1)).unsqueeze(0)).squeeze(0)
            # text_tensor = torch.tensor(tokenizer(current_node.text).input_ids , dtype=torch.float16).unsqueeze(0).cuda()
            print("text:{}".format(child.text))
            text_ids = tokenizer(child.text, padding=True, return_tensors='pt')
            for k, v in text_ids.items():
                text_ids[k] = v.cuda()
            outputs_2 = model(**text_ids).last_hidden_state
            text_embedding = pooling(outputs_2, text_ids, 'cls')
            # distance_text  = (x @ torch.nn.functional.normalize(text_embeddings, p=2, dim=0).permute(1, 0)).mean(0).mean(0) # remember to normalize the embedding
            # distance_image = (x @ torch.nn.functional.normalize(child.centroids.view(-1, query.size(-1)), p=2, dim=0).permute(1, 0)).mean(0).mean(0)
            sim = cos_sim(text_embedding[0], query_embedding[0])
            # current_node.text_distance = distance_text
            # current_node.image_distance = distance_image
            print(f"{RED}sim{RESET}:{sim} in depth:{current_node.depth} {i} child")
            # distance = (query @ child.centroids.view(-1, query.size(-1)).permute(1, 0)).sum(0).sum(0) # len x 1024 @ len x 1024 = len x len
            # min_distance, _ = distances.min(dim=0)
            if sim > best_sim:
                best_sim = sim
                best_child_index = i
            else:
                continue
            
        print(f"{BLUE}best_child_index:{RESET}{best_child_index}")
        # 选择相似度最高的子节点
        path_features.append(current_node.children[best_child_index].centroids)
        path_text.append(current_node.children[best_child_index].text)
        current_node = current_node.children[best_child_index]

    # path_features.append(current_node.centroids)  # 添加最底层节点的特征
    # path_text.append(current_node.text)

    return path_features, path_text

def fast_search_tree_multi_modal_with_embedding(all_nodes, query, image_embedding, model, tokenizer, top_k=1):
    """在树中沿着每一层相似度最高的分支不断寻找，并将每一层对应的特征都提取出来"""
    # The model works really well with cls pooling (default) but also with mean pooling.
    def pooling(outputs: torch.Tensor, inputs,  strategy: str = 'cls') -> np.ndarray:
        if strategy == 'cls':
            outputs = outputs[:, 0]
        elif strategy == 'mean':
            outputs = torch.sum(
                outputs * inputs["attention_mask"][:, :, None], dim=1) / torch.sum(inputs["attention_mask"])
        else:
            raise NotImplementedError
        return outputs.detach().cpu().numpy()
    
    path_features = []
    path_text = []
    redundant_nodes = []
    # x = torch.nn.functional.normalize(torch.cat([query, image_embedding], dim=0), p=2, dim=0)
    # print("Question:{}".format(query))
    
    query_ids = tokenizer(query, padding=True, return_tensors='pt')
    for k, v in query_ids.items():
        query_ids[k] = v.cuda()
    outputs_1 = model(**query_ids).last_hidden_state
    query_embedding = pooling(outputs_1, query_ids, 'cls')
    
    for node in all_nodes:
        current_node = node
        if current_node.depth == 0:
            redundant_nodes.append(node)                
        else:
            while current_node.children :
                # if current_node.
                best_child_index = None
                best_sim = 0
                
                # 计算查询张量与每个子节点的所有特征张量之间的相似度
                for i, child in enumerate(current_node.children):
                    # distances = torch.cdist(query.unsqueeze(0), child.centroids.view(-1, query.size(-1)).unsqueeze(0)).squeeze(0)
                    # text_tensor = torch.tensor(tokenizer(current_node.text).input_ids , dtype=torch.float16).unsqueeze(0).cuda()
                    print("text:{}".format(child.text))
                    text_ids = tokenizer(child.text, padding=True, return_tensors='pt')
                    for k, v in text_ids.items():
                        text_ids[k] = v.cuda()
                    outputs_2 = model(**text_ids).last_hidden_state
                    text_embedding = pooling(outputs_2, text_ids, 'cls')
                    # distance_text  = (x @ torch.nn.functional.normalize(text_embeddings, p=2, dim=0).permute(1, 0)).mean(0).mean(0) # remember to normalize the embedding
                    # distance_image = (x @ torch.nn.functional.normalize(child.centroids.view(-1, query.size(-1)), p=2, dim=0).permute(1, 0)).mean(0).mean(0)
                    sim = cos_sim(text_embedding[0], query_embedding[0])
                    # current_node.text_distance = distance_text
                    # current_node.image_distance = distance_image
                    print(f"{RED}sim{RESET}:{sim} in depth:{current_node.depth} {i} child")
                    # distance = (query @ child.centroids.view(-1, query.size(-1)).permute(1, 0)).sum(0).sum(0) # len x 1024 @ len x 1024 = len x len
                    # min_distance, _ = distances.min(dim=0)
                    if sim > best_sim:
                        best_sim = sim
                        best_child_index = i
                    else:
                        continue
                    
                print(f"{BLUE}best_child_index:{RESET}{best_child_index}")
                # 选择相似度最高的子节点
                path_features.append(current_node.children[best_child_index].centroids)
                path_text.append(current_node.children[best_child_index].text)
                current_node = current_node.children[best_child_index]
    
    # 针对存在的冗余节点再进行一次计算
    best_index = 0
    best_sim = 0
    
    if len(redundant_nodes) >= 1:
        for  i,node in enumerate(redundant_nodes):
            print("text:{}".format(node.text))
            text_ids = tokenizer(node.text, padding=True, return_tensors='pt')
            for k, v in text_ids.items():
                text_ids[k] = v.cuda()
            outputs_2 = model(**text_ids).last_hidden_state
            text_embedding = pooling(outputs_2, text_ids, 'cls')
            # distance_text  = (x @ torch.nn.functional.normalize(text_embeddings, p=2, dim=0).permute(1, 0)).mean(0).mean(0) # remember to normalize the embedding
            # distance_image = (x @ torch.nn.functional.normalize(child.centroids.view(-1, query.size(-1)), p=2, dim=0).permute(1, 0)).mean(0).mean(0)
            sim = cos_sim(text_embedding[0], query_embedding[0])
            # current_node.text_distance = distance_text
            # current_node.image_distance = distance_image
            print(f"{RED}sim{RESET}:{sim} in depth:{node.depth} {i} child")
            # distance = (query @ child.centroids.view(-1, query.size(-1)).permute(1, 0)).sum(0).sum(0) # len x 1024 @ len x 1024 = len x len
            # min_distance, _ = distances.min(dim=0)
            if sim > best_sim:
                best_sim = sim
                best_index = i
            else:
                continue
        print(f"{BLUE}best_redundant_index:{RESET}{best_index}")
        path_features.append(redundant_nodes[best_index].centroids)
        path_text.append(redundant_nodes[best_index].text)
    # else:
    #     # using the nearst feature 
    #     best_index = 0
        
    # print(f"{BLUE}best_redundant_index:{RESET}{best_index}")
    # path_features.append(redundant_nodes[best_index].centroids)
    # path_text.append(redundant_nodes[best_index].text)
    # path_features.append(current_node.centroids)  # 添加最底层节点的特征
    # path_text.append(current_node.text)

    return path_features, path_text

def fast_search_tree_multi_modal_with_embedding_not_repeat(all_nodes, query, image_embedding, model, tokenizer, top_k=1):
    """Search the tree by finding the branch with the highest similarity at each depth and extract the corresponding features."""

    def pooling(outputs: torch.Tensor, inputs, strategy: str = 'cls') -> np.ndarray:
        if strategy == 'cls':
            outputs = outputs[:, 0]
        elif strategy == 'mean':
            outputs = torch.sum(
                outputs * inputs["attention_mask"][:, :, None], dim=1) / torch.sum(inputs["attention_mask"])
        else:
            raise NotImplementedError
        return outputs.detach().cpu().numpy()

    def get_query_embedding(query, tokenizer, model):
        query_ids = tokenizer(query, padding=True, return_tensors='pt')
        for k, v in query_ids.items():
            query_ids[k] = v.cuda()
        outputs_1 = model(**query_ids).last_hidden_state
        return pooling(outputs_1, query_ids, 'cls')

    def get_text_embedding(text, tokenizer, model):
        text_ids = tokenizer(text, padding=True, return_tensors='pt')
        for k, v in text_ids.items():
            text_ids[k] = v.cuda()
        outputs_2 = model(**text_ids).last_hidden_state
        return pooling(outputs_2, text_ids, 'cls')

    path_features = []
    path_text = []

    # Compute the query embedding once
    query_embedding = get_query_embedding(query, tokenizer, model)

    # Step 1: Find the most similar node with depth > 0
    best_node_index = None
    best_node_sim = 0
    best_node_depth = 0

    for i, node in enumerate(all_nodes):
        if node.depth > 0:  # Consider nodes with depth > 0
            text_embedding = get_text_embedding(node.text, tokenizer, model)
            sim = cos_sim(text_embedding[0], query_embedding[0])

            if sim > best_node_sim:
                best_node_sim = sim
                best_node_index = i
                best_node_depth = node.depth

    if best_node_index is None:
        print("No nodes with depth > 0 found.")
        return path_features, path_text

    # Use the most similar node to search its children
    current_node = all_nodes[best_node_index]
    print(f"Best node index: {best_node_index}, similarity: {best_node_sim}, depth: {best_node_depth}")

    while current_node.children:
        best_child_index = None
        best_child_sim = 0

        for i, child in enumerate(current_node.children):
            text_embedding = get_text_embedding(child.text, tokenizer, model)
            sim = cos_sim(text_embedding[0], query_embedding[0])

            if sim > best_child_sim:
                best_child_sim = sim
                best_child_index = i

        if best_child_index is not None:
            print(f"Best child index: {best_child_index}, similarity: {best_child_sim}")
            path_features.append(current_node.children[best_child_index].centroids)
            path_text.append(current_node.children[best_child_index].text)
            current_node = current_node.children[best_child_index]
        else:
            break

    return path_features, path_text

def search_tree_multi_modal(node, query, image_embedding, model, tokenizer, top_k=1):
    """在树中沿着每一层相似度最高的分支不断寻找，并将每一层对应的特征都提取出来"""
    path_features = []
    path_text = []
    x = torch.nn.functional.normalize(torch.cat([query, image_embedding], dim=0), p=2, dim=0)
    current_node = node
    while current_node.children:
        best_child_index = None
        best_distance = float('inf')
        
        # 计算查询张量与每个子节点的所有特征张量之间的相似度
        for i, child in enumerate(current_node.children):
            # distances = torch.cdist(query.unsqueeze(0), child.centroids.view(-1, query.size(-1)).unsqueeze(0)).squeeze(0)
            # text_tensor = torch.tensor(tokenizer(current_node.text).input_ids , dtype=torch.float16).unsqueeze(0).cuda()
            text_ids = tokenizer(current_node.text).input_ids
            text_embeddings  = model.get_model().embed_tokens(torch.tensor(text_ids, dtype=torch.long, device='cuda')) # num_text_token 4096
            distance_text  = (x @ torch.nn.functional.normalize(text_embeddings, p=2, dim=0).permute(1, 0)).mean(0).mean(0) # remember to normalize the embedding
            distance_image = (x @ torch.nn.functional.normalize(child.centroids.view(-1, query.size(-1)), p=2, dim=0).permute(1, 0)).mean(0).mean(0)
            distance = 0.5*distance_text + 0.1*distance_image
            # current_node.text_distance = distance_text
            # current_node.image_distance = distance_image
            print("distance:{}/{}".format(distance_text, distance_image))
            # distance = (query @ child.centroids.view(-1, query.size(-1)).permute(1, 0)).sum(0).sum(0) # len x 1024 @ len x 1024 = len x len
            # min_distance, _ = distances.min(dim=0)
            if distance < best_distance:
                best_distance = distance
                best_child_index = i
            else:
                best_child_index = i
                continue

        # 选择相似度最高的子节点
        path_features.append(current_node.centroids)
        path_text.append(current_node.text)
        current_node = current_node.children[best_child_index]

    path_features.append(current_node.centroids)  # 添加最底层节点的特征
    path_text.append(current_node.text)

    return path_features, path_text

def search_tree(node, query, top_k=1):
    """在树中沿着每一层相似度最高的分支不断寻找，并将每一层对应的特征都提取出来"""
    path_features = []

    current_node = node
    while current_node.children:
        best_child_index = None
        best_distance = float('inf')
        
        # 计算查询张量与每个子节点的所有特征张量之间的相似度
        for i, child in enumerate(current_node.children):
            # distances = torch.cdist(query.unsqueeze(0), child.centroids.view(-1, query.size(-1)).unsqueeze(0)).squeeze(0)
            distance = (query @ child.centroids.view(-1, query.size(-1)).permute(1, 0)).sum(0).sum(0) # len x 1024 @ len x 1024 = len x len
            # min_distance, _ = distances.min(dim=0)
            if distance < best_distance:
                best_distance = distance
                best_child_index = i
            else:
                best_child_index = i

        # 选择相似度最高的子节点
        path_features.append(current_node.centroids)
        current_node = current_node.children[best_child_index]

    path_features.append(current_node.centroids)  # 添加最底层节点的特征

    return path_features


def long_short_memory_update(feature_bank, short_window=10, remember_window=4, tau=10, compress_rate=2,  chunk_size=100, num_clusters=10, interval=3 ):
    """
    this fucntion is used for updating the long short memory buffer
    """
    ############# building short memory ###################
    
    assert len(feature_bank) > short_window
    waite_FIFO = feature_bank[-short_window:]
    assert len(waite_FIFO) > remember_window
    forgetting_probs = calculate_forgetting_probabilities(short_window, tau=tau)
    short_memory_buffer = select_data_without_replacement(waite_FIFO, forgetting_probs, remember_window)
    
    ############# building long memory ###################
    if compress_rate > 1:
        compressed_spatial_feature_list = compress_spatial_features(feature_bank, compress_rate)
    else:
        compressed_spatial_feature_list = feature_bank
    chunk_feature_list = [compressed_spatial_feature_list[i:i + chunk_size] for i in range(0, len(compressed_spatial_feature_list), chunk_size)] # length100 
    k_means_chunk_feature_list = [weighted_kmeans_feature(torch.cat(chunk_feature), num_clusters)[0] if len(chunk_feature)> 10 else torch.cat(chunk_feature) for chunk_feature in chunk_feature_list] # length100 最后一个不需要聚类
    
    long_memory_tree = buildingd_memory_tree_buttom_up(k_means_chunk_feature_list, num_clusters, interval)
    
    # print("time spend:{}/{}/{}".format((time_2-time_1), (time_3-time_2), (time_4-time_3)))
    return short_memory_buffer, long_memory_tree

def long_short_memory_update_with_summarize(feature_bank, summarizer, tokenizer, input_ids,  short_window=10, remember_window=4, tau=10, compress_rate=2,  chunk_size=100, num_clusters=10, interval=3 ):
    """
    this fucntion is used for updating the long short memory buffer
    """
    def print_tree(node, indent=0):
        """打印树状结构"""
        print(" " * indent + f"{RED}Depth{RESET}: {node.depth}, {RED}Centroids{RESET}: {node.centroids.shape}, {RED}Text{RESET} :{node.text}")
        for child in node.children:
            print_tree(child, indent + 4)
    
    ############# building short memory ###################
    print("<<<<<<< building short memory >>>>>>>>>>>")
    if len(feature_bank) > short_window:
    # assert len(feature_bank) > short_window
        waite_FIFO = feature_bank[-short_window:]
    else:
        short_window = len(feature_bank)
        waite_FIFO = feature_bank
    # assert len(waite_FIFO) > remember_window
    if remember_window > len(waite_FIFO):
        remember_window = len(waite_FIFO)
        
    forgetting_probs = calculate_forgetting_probabilities(short_window, tau=tau)
    short_memory_buffer = select_data_without_replacement(waite_FIFO, forgetting_probs, remember_window)
    
    ############# building long memory with image captioning ###################
    
    if compress_rate > 1:
        compressed_spatial_feature_list = compress_spatial_features(feature_bank, compress_rate) # len 
    else:
        compressed_spatial_feature_list = feature_bank
    chunk_feature_list = [compressed_spatial_feature_list[i:i + chunk_size] for i in range(0, len(compressed_spatial_feature_list), chunk_size)] # length100 
    k_means_chunk_feature_list = [weighted_kmeans_feature(torch.cat(chunk_feature), num_clusters)[0] if len(chunk_feature)> chunk_size else torch.cat(chunk_feature) for chunk_feature in chunk_feature_list] # length100 最后一个不需要聚类
    print("<<<<<<< building long memory tree >>>>>>>>>>>")
    long_memory_tree = buildingd_memory_tree_buttom_up_with_summarize_token(k_means_chunk_feature_list, num_clusters, interval, summarizer, input_ids, tokenizer, chunk_feature_list)
    print_tree(long_memory_tree)
    # print("time spend:{}/{}/{}".format((time_2-time_1), (time_3-time_2), (time_4-time_3)))
    return short_memory_buffer, long_memory_tree

# def real_time_long_short_memory_update_with_summarize(feature_bank, summarizer, tokenizer, input_ids,  short_window=10, remember_window=4, tau=10, compress_rate=2,  chunk_size=100, num_clusters=10, interval=3 ):
#     """
#     this fucntion is used for updating the long short memory buffer
#     """
#     def print_tree(node, indent=0):
#         """打印树状结构"""
#         print(" " * indent + f"{RED}Depth{RESET}: {node.depth}, {RED}Centroids{RESET}: {node.centroids.shape}, {RED}Text{RESET} :{node.text}")
#         for child in node.children:
#             print_tree(child, indent + 4)
    
#     ############# building short memory ###################
#     print("<<<<<<< building short memory >>>>>>>>>>>")
#     if len(feature_bank) > short_window:
#     # assert len(feature_bank) > short_window
#         waite_FIFO = feature_bank[-short_window:]
#     else:
#         short_window = len(feature_bank)
#         waite_FIFO = feature_bank
#     # assert len(waite_FIFO) > remember_window
#     if remember_window > len(waite_FIFO):
#         remember_window = len(waite_FIFO)
        
#     forgetting_probs = calculate_forgetting_probabilities(short_window, tau=tau)
#     short_memory_buffer = select_data_without_replacement(waite_FIFO, forgetting_probs, remember_window)
    
#     ############# building long memory with image captioning ###################
    
#     if compress_rate > 1:
#         compressed_spatial_feature_list = compress_spatial_features(feature_bank, compress_rate) # len 
#     else:
#         compressed_spatial_feature_list = feature_bank
#     chunk_feature_list = [compressed_spatial_feature_list[i:i + chunk_size] for i in range(0, len(compressed_spatial_feature_list), chunk_size)] # length100 
#     k_means_chunk_feature_list = [weighted_kmeans_feature(torch.cat(chunk_feature), num_clusters)[0] if len(chunk_feature)> chunk_size else torch.cat(chunk_feature) for chunk_feature in chunk_feature_list] # length100 最后一个不需要聚类
#     print("<<<<<<< building long memory tree >>>>>>>>>>>")
#     long_memory_tree = fast_building_memory_tree_summarize_token(k_means_chunk_feature_list, num_clusters, interval, summarizer, input_ids, tokenizer, chunk_feature_list)
#     # print_tree(long_memory_tree)
#     # print("time spend:{}/{}/{}".format((time_2-time_1), (time_3-time_2), (time_4-time_3)))
#     return short_memory_buffer, long_memory_tree

def count_nodes_by_depth(nodes):
    depth_count = defaultdict(int)
    for node in nodes:
        depth_count[node.depth] += 1
        stack = node.children.copy()
        while stack:
            current_node = stack.pop()
            depth_count[current_node.depth] += 1
            stack.extend(current_node.children)
    return depth_count
    
def build_prompt_with_search_memory(history, text, user_memory, user_name, user_memory_index, local_memory_qa, meta_prompt, new_user_meta_prompt, user_keyword, ai_keyword, boot_actual_name, language):
    # history_content = ''
    # for query, response in history:
    #     history_content += f"\n [|用户|]：{query}"
    #     history_content += f"\n [|AI伴侣|]：{response}"
    # history_content += f"\n [|用户|]：{text} \n [|AI伴侣|]："
    memory_search_query = text#f'和对话历史：{history_content}。最相关的内容是？'
    memory_search_query = memory_search_query.replace(user_keyword,user_name).replace(ai_keyword,'AI')
    if user_memory_index:
        related_memos, memo_dates= local_memory_qa.search_memory(memory_search_query,user_memory_index)
        related_memos = '\n'.join(related_memos)
    else:
        related_memos = ""
    
    # print("related_memos",related_memos)
    
    # print("user_memory", user_memory) # always get memory
    if "overall_history" in user_memory:
        history_summary = "你和用户过去的回忆总结是：{overall}".format(overall=user_memory["overall_history"]) if language=='cn' else "The summary of your past memories with the user is: {overall}".format(overall=user_memory["overall_history"])
    else:
        history_summary = ''
    # mem_summary = [(k, v) for k, v in user_memory['summary'].items()]
    # memory_content += "最近的一段回忆是：日期{day}的对话内容为{recent}".format(day=mem_summary[-1][0],recent=mem_summary[-1][1])
    related_memory_content = f"\n{str(related_memos).strip()}\n"
    personality = user_memory['overall_personality'] if "overall_personality" in user_memory else ""
   
    # print("personality",personality)
    history_text = ''
    for dialog in history:
        query = dialog['query']
        response = dialog['response']
        history_text += f"\n {user_keyword}: {query}"
        history_text += f"\n {ai_keyword}: {response}"
    history_text += f"\n {user_keyword}: {text} \n {ai_keyword}: " 
    
    if history_summary and related_memory_content and personality:
        prompt = meta_prompt.format(user_name=user_name,history_summary=history_summary,related_memory_content=related_memory_content,personality=personality,boot_actual_name=boot_actual_name,history_text=history_text,memo_dates=memo_dates)
    elif related_memory_content:
        prompt = meta_prompt.format(user_name=user_name,related_memory_content=related_memory_content,boot_actual_name=boot_actual_name,memo_dates=memo_dates)
    else:
        prompt = new_user_meta_prompt.format(user_name=user_name,boot_actual_name=boot_actual_name,history_text=history_text)
    # print(prompt)
    return prompt

def build_prompt_with_search_memory_only_related(text, user_name, user_memory_index, local_memory_qa, meta_prompt, user_keyword, ai_keyword, boot_actual_name):
    memory_search_query = text
    memory_search_query = memory_search_query.replace(user_keyword,user_name).replace(ai_keyword,'AI')
    # memory_search_query = memory_search_query.replace(user_keyword,user_name)
    if user_memory_index:
        related_memos, memo_dates= local_memory_qa.search_memory(memory_search_query,user_memory_index)
        related_memos = '\n'.join(related_memos)
        related_memory_content = f"\n{str(related_memos).strip()}\n"
        # related_memory_content = f"\n{str(related_memos).strip()}\n".replace('AI','LLaVA')
    else:
        related_memos = None
        related_memory_content = None
    
    if related_memory_content is not None:
        # print("related_memory_content",related_memory_content)
        # prompt = meta_prompt.format(user_name=user_name,related_memory_content=related_memory_content,boot_actual_name=boot_actual_name)
        prompt = meta_prompt.format(related_memory_content=related_memory_content)
    else:
        # prompt = new_user_meta_prompt.format(user_name=user_name,boot_actual_name=boot_actual_name,history_text=None)
        prompt = None
    # print(prompt)
    return prompt

def convert_to_markdown(text):
    text = text.replace("$", "&#36;")

    def replace_leading_tabs_and_spaces(line):
        new_line = []

        for char in line:
            if char == "\t":
                new_line.append("&#9;")
            elif char == " ":
                new_line.append("&nbsp;")
            else:
                break
        return "".join(new_line) + line[len(new_line) :]

    markdown_text = ""
    lines = text.split("\n")
    in_code_block = False

    for line in lines:
        if in_code_block is False and line.startswith("```"):
            in_code_block = True
            markdown_text += "```\n"
        elif in_code_block is True and line.startswith("```"):
            in_code_block = False
            markdown_text += "```\n"
        elif in_code_block:
            markdown_text += f"{line}\n"
        else:
            line = replace_leading_tabs_and_spaces(line)
            line = re.sub(r"^(#)", r"\\\1", line)
            markdown_text += f"{line}  \n"

    return markdown_text

def visualize_memory_feature_with_PCA(all_feature_list, clustering_feature_list, clustering, same_color = True, only_most_important = True):
    pca = PCA(n_components=2)
    # 对主成分进行归一化
    scaler = MinMaxScaler()
    
    
    # print("all feature list :{}/{}".format(len(all_feature_list), all_feature_list[0].shape))
    # print("clustering feature list :{}/{}".format(len(clustering_feature_list), clustering_feature_list[0].shape))
    
    all_features= torch.cat(all_feature_list, dim=0) # 
    all_features = torch.sum(all_features, dim=1)
    all_features = all_features.view(-1, all_features.shape[-1]).detach().cpu()
    
    clustering_features = torch.cat(clustering_feature_list, dim=0)
    clustering_features = torch.sum(clustering_features, dim=1) # num_
    clustering_features = clustering_features.view(-1, clustering_features.shape[-1]).detach().cpu()
    
    # 定义颜色列表
    colors = ['red', 'orange', 'green', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
    
    color_for_original = 'blue'
    newX = pca.fit_transform(all_features)     #等价于pca.fit(X) pca.transform(X)
    # invX = pca.inverse_transform(newX)  #将降维后的数据转换成原始数据
    print("newX:{}".format(newX.shape))
    # print("invX:{}".format(invX.shape))
    
    newC = pca.fit_transform(clustering_features)     #等价于pca.fit(X) pca.transform(X)
    # invX = pca.inverse_transform(newX)  #将降维后的数据转换成原始数据
    # print("invX:{}".format(invX.shape))

    # tsne=TSNE()
    # all_features_tsne = tsne.fit_transform(all_features)  #进行数据降维,降成两维
    # print("all feature shape:{}".format(all_features_tsne.shape))
    
    # clustering_features_tsne = tsne.fit_transform(clustering_features)  #进行数据降维,降成两维
    # print("clustering features shape:{}".format(clustering_features_tsne.shape))
    
    #a=tsne.fit_transform(data_zs) #a是一个array,a相当于下面的tsne_embedding

    
    # normalize 
    normalized_newX = scaler.fit_transform(newX)
    normalized_newC = scaler.fit_transform(newC)
    if only_most_important:
        normalized_newC = normalized_newC[:clustering, :]
        
    print("normalized_newC:{}".format(normalized_newC.shape)) # 降维后的聚类数据
    
    # 分配颜色
    color_map = [colors[(i // clustering) % len(colors)] for i in range(len(normalized_newC))]
    
    # d=tsne[r[u'聚类类别']==0]
    # plt.plot(normalized_newX[:,0],normalized_newX[:,1],'b.')
    plt.scatter(normalized_newX[:,0],normalized_newX[:,1],c=color_map, s=10)

    # d=tsne[r[u'聚类类别']==1]
    # plt.plot(d[0],d[1],'go')

    # d=tsne[r[u'聚类类别']==2]
    if same_color:
        plt.plot(normalized_newC[:,0],normalized_newC[:,1],'r*')
    elif not same_color:
        plt.scatter(normalized_newC[:,0],normalized_newC[:,1],c=color_map, s=10)
        
    plt.show()
    plt.savefig("./PCA_long_memory.jpg", dpi=300)
    # plt.savefig('pca_normalized.png', dpi=300)

def embedding_pca(embeddings, n_components=3, as_rgb=True):
    '''
    输入:
    embeddings: 网络的特征，维度为[C, H, W]
    n_components: 将网络的特征降维成多少个通道，默认为3，即图片的RGB三个通道
    as_rgb: 是否转换为图片格式，默认为是

    输出:
    embed_flat: 一个通道数量为3的图片矩阵，维度为[H, W, 3]
    函数返回的结果，可以通过cv2等工具直接保存为图片
    例如: cv2.imwrite('pca.png', embed_flat)
    '''

    pca = PCA(n_components=n_components)
    patch_size = int(math.sqrt(embeddings.shape[1]))
    embeddings = embeddings.reshape(-1, patch_size, patch_size, -1).squeeze(0)
    embed_dim = embeddings.shape[0]
    shape = embeddings.shape[1:]

    embed_flat = embeddings.reshape(embed_dim, -1).T
    embed_flat = pca.fit_transform(embed_flat).T
    embed_flat = embed_flat.reshape((n_components,) + shape)

    if as_rgb:
        embed_flat = 255 * (embed_flat - embed_flat.min()) / np.ptp(embed_flat)
        embed_flat = np.transpose(embed_flat, (1,2,0))
        embed_flat = embed_flat.astype('uint8')
    return embed_flat

def test_2(feature_list):
    chunk_size = 100
    
    time_1 = time.time()
    
    feature_list = compress_spatial_features(feature_list, 2)
    
    time_2 = time.time()
    
    chunk_feature_list = [feature_list[i:i + chunk_size] for i in range(0, len(feature_list), chunk_size)] # length100 
    
    k_means_chunk_feature_list = [weighted_kmeans_feature(torch.cat(chunk_feature), 10)[0] if len(chunk_feature)> 10 else chunk_feature for chunk_feature in chunk_feature_list] # length100 最后一个不需要聚类
    print("k_means_chunk_feature_list",k_means_chunk_feature_list[0].shape)
    time_3 = time.time()

    root = buildingd_memory_tree_buttom_up(k_means_chunk_feature_list, 10, 3)
    
    time_4 = time.time()
    
    print("time spend:{}/{}/{}".format((time_2-time_1), (time_3-time_2), (time_4-time_3)))
    return root

def test_3(feature_list):
    
    short_memory_buffer, long_memory_tree = long_short_memory_update(feature_list, short_window=8, 
                                                                     remember_window=4, tau=10, 
                                                                     compress_rate=2, chunk_size=50, 
                                                                     num_clusters=5, interval=2)
    # utilized in system
    # (cache, short_window=8, 
    # remember_window=4, tau=10, 
    # compress_rate=2, chunk_size=50, 
    # num_clusters=5, interval=2)
    
    # used for test
    # (cache, short_window=8, 
    # remember_window=4, tau=10, 
    # compress_rate=2, chunk_size=10, 
    # num_clusters=10, interval=5)
    
    return short_memory_buffer, long_memory_tree

def test_memory():
    
    conversation = [
        {
            "LLaVA":"The image appears to be a screenshot from a video, showing a person wearing parachuting gear and an airplane helmet with a camera attached. The person is in an airplane that is flying at a high altitude, as indicated by the curvature of the Earth visible behind them. The background features the sky with some cloud cover, and there's a trail that might be a road or path below, suggesting that the airplane is flying above a countryside area with greenery. The motion blur and distortion indicate that the image might have been taken with a camera mounted on the helmet, which is a common practice for recording first-person views during adventurous activities such as skydiving, bungee jumping, or air racing. The letter 'B' in the corner suggests this could be from a video series or a platform that uses 'B' to denote specific content.",
            "User":"please tell me what you see from these images?"
        },
        {
            "LLaVA":"Based on the information you provided, I deduced that this video was showing a fighter pilot performing a mission, I saw that one of the images said 'setting time to 2 minutes 15 seconds', all I guessed was the time of the mission The limit is 2 minutes and 15 seconds.",
            "User":"I saw the pilot was performing a mission, what is the time limit for the mission?"
        },
        {
            "LLaVA":"In the image you provided, the person shown appears to be piloting a fighter jet over a mountainous terrain. However, without additional context, it's not possible to confirm if the individual is Maverick from the film series 'Top Gun.' Maverick, the main character played by Tom Cruise, is a fictional character and is not an identifiable real person. This image could depict a stunt pilot, a character from a specific film or video game, or could be an actual person involved in an air show or military exercise. To determine if the pilot is Maverick specifically, one would need to identify distinctive features from the film series' storyline or recognizable elements from the pilot's character design.",
            "User":"Is mavrrick the pilot in the film?"
        },
        {
            "LLaVA":"I think it,s 2 minutes 15 seconds.",
            "User":"Now please repeat what is the time limit for the task."
        },
        {
            "LLaVA":"The image appears to show a person piloting a glider or parachute, with a high-speed action blur in the background, creating a sense of motion. The blurriness suggests that the person is traveling at high speeds, which is a common experience in activities like skydiving or hang-gliding. The person is wearing a helmet with goggles or a visor, and there are straps around their neck, likely part of the harness system used for safety and control. The background features a landscape that looks like it could be a rural area with a winding road or path leading through it. There's also a logo or symbol in the bottom left corner, which looks like a stylized 'B', but without additional context, it's not clear what it represents.",
            "User":"please tell me what you see from this video?"
        }
    ]
    # print("Building LLaMA3 clinet !!!")
    # config = {'max_tokens':1024, 'temperature':1, 'top_p':0.95, 'frequency_penalty':True}
    # llm_client = LLMClientLLaMA3(gen_config=config, model_name='/13390024681/All_Model_Zoo/llama3-8b-instruct-hf')
    # print("LLaMA3 Client Building Finish !!!")

    data_args,model_args = HfArgumentParser(
        (DataArguments,ModelArguments)
    ).parse_args_into_dataclasses()
    
    memory_dir = os.path.join(data_args.memory_basic_dir,data_args.memory_file)
    print(memory_dir)
    if not os.path.exists(memory_dir):
        json.dump({},open(memory_dir,"w",encoding="utf-8"))

    language = data_args.language
    enable_forget_mechanism = data_args.enable_forget_mechanism
    print(language, enable_forget_mechanism, memory_dir)
    
    if data_args.enable_forget_mechanism:
        from memory_bank.memory_retrieval.forget_memory_new import LocalMemoryRetrieval
    else:
        from memory_retrieval.local_doc_qa import LocalMemoryRetrieval
        
    local_memory_qa = LocalMemoryRetrieval()
    # Embedding model name
    EMBEDDING_MODEL_CN = "text2vec"
    EMBEDDING_MODEL_EN = "minilm-l6"
    EMBEDDING_MODEL = EMBEDDING_MODEL_CN if language == 'cn' else EMBEDDING_MODEL_EN
    local_memory_qa.init_cfg(
                            embedding_model=EMBEDDING_MODEL,
                            embedding_device="cuda",
                            top_k=1,
                            language=language)

    meta_prompt = generate_meta_prompt_dict_chatglm_app()[language]
    new_user_meta_prompt = generate_new_user_meta_prompt_dict_chatglm()[language]
    
    only_related_prompt = only_related_prompt_dict_llava_app()[language]
    
    user_keyword = '[|User|]'
    ai_keyword = '[|LLaVA|]'
    
    # boot_name = boot_name_dict[language]
    boot_actual_name = "LLaVA"
    
    
    # memory_dir = '/13390024681/llama/EfficientVideo/Ours/memory_bank/memories.json'
    memory = json.loads(open(memory_dir,"r",encoding="utf-8").read())
    print('Please Enter Your Name:')
    user_name = input("\nUser name：")
    # print(memory.keys())
    if user_name in memory.keys():
        if input('Would you like to summarize your memory? If yes, please enter "yes"') == "yes":
            print("Building ChtaGLM clinet !!!")
            config = {'max_tokens':1024, 'temperature':1, 'top_p':0.95, 'frequency_penalty':True}
            llm_client = LLMClientLLaMA3(gen_config=config, model_name='/13390024681/All_Model_Zoo/chatglm3-6b')
            print("ChtaGLM Client Building Finish !!!")
            user_memory = summarize_memory_event_personality(data_args, memory, user_name, llm_client)
    hello_msg,user_memory,memory,user_name,user_memory_index = enter_name(user_name,memory,local_memory_qa,data_args)
    print(hello_msg, user_memory_index)
    # assert 1==2
    
    history = []
    
    for conv in conversation:
        # query = "I just wann test the memory retrival function !!"
        query = conv['User']
        robot_answer = conv['LLaVA']
        
        if len(history) > data_args.max_history:
            history = history[-data_args.max_history:]

        print("QUESTION:{}".format(query))
        prompt = build_prompt_with_search_memory_only_related(query, user_name, user_memory_index, local_memory_qa, only_related_prompt, user_keyword, ai_keyword, boot_actual_name)
        print("PROMPT:{}".format(prompt))
        response = robot_answer
        
        result = response

        torch.cuda.empty_cache()
    
        # a, b = [[y[0], convert_to_markdown(y[1])] for y in history] + [[query, convert_to_markdown(result)]], history + [[query, result]]
        b = [[query, result]]
        print("b",b)
        # a, b = [[y[0], convert_to_markdown(y[1])] for y in history] ,history 
        if user_name:
            memory = save_local_memory(memory,b,user_name,data_args)
        
        #  update information
        _,_,memory,user_name,user_memory_index = enter_name(user_name,memory,local_memory_qa,data_args)
        
        time.sleep(2)
    pass

def prepare_input_for_loss():
    pass

def test_eval_metrics():
    from llavanext.model.builder import load_pretrained_model
    from llavanext.utils import disable_torch_init
    from llava.eval.model_utils import load_video
    from llavanext.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
    from llavanext.conversation import conv_templates, SeparatorStyle
    from llavanext.mm_utils import expand2square

    question_list = [
        {
            "question": "This is a clip from the movie 'Truman'. Please describe the appearance of the ship in this video clip.",
            "answer": "Truman's boat is a small white sailboat with wooden masts and white sails.",
            "class": "describe",
            "time": 240
        },
        {
            "question": "What happened to the boat shown in the picture?",
            "answer": "The boat hit a wall while sailing, making it impossible to continue moving forward.",
            "class": "describe",
            "time": 1320
        },
        {
            "question": "Why does Truman keep hitting the wall?",
            "answer":"The boat he was driving crashed into a wall, causing him to become stranded at sea, leaving him furious and frustrated.",
            "class": "describe",
            "time": 1440
        },
        {
            "question": "How do Truman feel and what's the reason ?",
            "answer":"Truman feels very frustrated. From the previous information, I learned that his boat was stuck on the sea and could not continue sailing.",
            "class": "describe",
            "time": 2232
        },
        {
            "question": "What is Truman wearing black shirt doing?",
            "answer":"Truman seemed to have found a way out, and he was following it.",
            "class": "describe",
            "time": 3120
        },
        {
            "question": "Why does Truman climb the steps?",
            "answer":"Before, he was stuck at sea, now he found a way to the exit door, so he climbed the steps towards that door.",
            "class": "describe",
            "time": 3600
        },
        {
            "question": "Now what are the characteristics of this man you see, is he the man trapped at sea? ",
            "answer":"The man wears a brown hat and glasses, but he is not a man trapped on the sea.",
            "class": "describe",
            "time": 5280
        },
        {
            "question": "According to the scene you saw, did Truman trapped on the sea find a way out?",
            "answer":"Yes, he find the door to leave, and he is standing in front of that door.",
            "class": "describe",
            "time": 6000
        },
        {
            "question": "What are people in bar doing?",
            "answer":"These people were watching TV, and what was playing on the TV was Truman's live broadcast.",
            "class": "describe",
            "time": 7608
        },
        {
            "question": "Why is this man watching Truman Live while taking a bath in the bathtub so happy?",
            "answer":"Because he is also happy that Truman find a way to leave.",
            "class": "describe",
            "time": 8880
        },
        {
            "question": "How many women are there on the table watching Truman Live?",
            "answer":"There are two women。",
            "class": "describe",
            "time": 9120
        }
    ]
    
    disable_torch_init()
    
    model_path = '/13390024681/All_Model_Zoo/LLaVA-NExT-LLaMA3-8B'
    model_base = None
    model_name = 'llava_next_llama3'
    conv_mode = 'llava_llama_3'
    video_path = '/13390024681/llama/EfficientVideo/Ours/videos/1.mp4'
    print("Loading MOdels for metric measure !!!")
    
    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, model_base, model_name)
    model.eval()
    
    print("Model Loading Finished !!")
    
    print("Loading Video from path !!")
    
    def llava_inference_for_loss(video_frames, question,labels, conv_mode, model, tokenizer, image_processor, image_sizes):
        if model.config.mm_use_im_start_end:
            qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + question + labels
        else:
            qs = DEFAULT_IMAGE_TOKEN + '\n' + question + labels

        conv = conv_templates[conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        # print(question)
        input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt')
        input_ids = input_ids.unsqueeze(0).cuda()
        # ques_ids = ques_ids.unsqueeze(0).cuda()
        labels = input_ids
        # labels = tokenizer(labels).input_ids
        # labels = torch.tensor(labels, dtype=torch.long).unsqueeze(0).cuda()
        
        # input_ids = torch.cat([])
        
        print("input ids:{} label:{}".format(input_ids.shape, labels.shape))
        attention_mask = None
        position_ids   = None
        past_key_values = None
        inputs_embeds = None
        # print("input",input_ids)
        # print("ques",ques_ids)
        
        # image_tensor = process_images(video_frames, image_processor, model.config) 
        new_images = []
        image_features = []
        
        for image in video_frames:
            image = expand2square(image, tuple(int(x * 255) for x in image_processor.image_mean))
            image = image_processor.preprocess(image, return_tensors="pt")["pixel_values"][0]
            new_images.append(image.to(dtype=torch.float16, device='cuda', non_blocking=True)) # [ dim resize_w resize_h ]

        for image_tensor in new_images:
            image_embedding = model.only_encode(image_tensor.unsqueeze(0).to(dtype=torch.float16))
            # print("image embedding without proj:{}".format(image_embedding.shape)) # 1 576 1024
            image_embedding = model.only_project(image_embedding)
            image_features.append(image_embedding)
        
        print("iimage_features:{}".format(image_features[0].shape))
        
        images = [torch.cat(image_features).view(-1, image_features[0].shape[-1])]
        
        with torch.no_grad():
            
            output = model.forward_with_fix_embedding(
                input_ids,
                attention_mask = attention_mask,
                position_ids = position_ids,
                past_key_values = past_key_values,
                inputs_embeds = inputs_embeds,
                labels = labels,
                images=images,
                image_sizes=image_sizes,
                return_dict = True)

        # outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
        return output
    
    video_frames, sizes = load_video(video_path, num_frm=6)
    
    all_ppl = []
    for data in question_list:
        question = data['question']
        labels = data['answer']
        
        output = llava_inference_for_loss(video_frames, question, labels, conv_mode, model, tokenizer, image_processor, sizes)
        loss = output.loss
        # print(loss)
        ppl = torch.exp(loss)
        # print(ppl)
        all_ppl.append(ppl.detach().cpu().numpy())
        # assert 1==2

    print(all_ppl)
    
    print("Avg PPL:",sum(all_ppl)/len(all_ppl))
    # 计算相邻两个数据之间的差异
    differences = [abs(all_ppl[i+1] - all_ppl[i]) for i in range(len(all_ppl) - 1)]
    
    # 求差异的总和
    total_difference = sum(differences)
    
    # 计算差异的数量
    count_differences = len(differences)
    
    # 计算差异的平均值
    average_difference = total_difference / count_differences
    
    print("Fluency PPL:", average_difference)

def test_multi_modal_memory_tree():
    
    # from 
    pass

def test_embedding():
    from typing import Dict

    import torch
    import numpy as np
    from transformers import AutoModel, AutoTokenizer
    from sentence_transformers.util import cos_sim

    # For retrieval you need to pass this prompt. Please find our more in our blog post.
    def transform_query(query: str) -> str:
        """ For retrieval, add the prompt for query (not for documents).
        """
        return f'Represent this sentence for searching relevant passages: {query}'

    # The model works really well with cls pooling (default) but also with mean pooling.
    def pooling(outputs: torch.Tensor, inputs: Dict,  strategy: str = 'cls') -> np.ndarray:
        if strategy == 'cls':
            outputs = outputs[:, 0]
        elif strategy == 'mean':
            outputs = torch.sum(
                outputs * inputs["attention_mask"][:, :, None], dim=1) / torch.sum(inputs["attention_mask"])
        else:
            raise NotImplementedError
        return outputs.detach().cpu().numpy()

    # 1. load model
    model_id = '/13390024681/All_Model_Zoo/mxbai-colbert-large-v1'
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModel.from_pretrained(model_id).cuda()


    # docs = [
    #     transform_query('A man is eating a piece of bread'),
    #     "A man is eating food.",
    #     "A man is eating pasta.",
    #     "The girl is carrying a baby.",
    #     "A man is riding a horse.",
    # ]
    # docs = [
    #     transform_query('A man is eating a piece of bread'),
    #     "A man is eating food.",
    # ]
    
    # x_1  = transform_query('A man is eating a piece of bread')
    # x_1 = "The video captures a scene of a room undergoing renovation or construction. The floor is made of hardwood, and the space appears to be spacious with a circular layout that suggests it might be part of an open-concept design.\
    #     In the foreground, there's a person who seems to be in the midst of work. They are wearing a navy blue t-shirt with white text that reads 'NECHES' and additional smaller text below it. The individual is holding what appears to be a long, flat piece of material, possibly a piece of wood or a composite board, which they seem to be measuring or aligning with something else on the floor.\
    #     The room is well-lit, and various tools and equipment are scattered around, indicating active work. There are power drills, drills with bits attached, and other hand tools like hammers and screwdrivers. There are also rolls of cable or extension cords running across the floor, suggesting electrical work being done.\
    #     In the background, there's more evidence of the ongoing work: partially installed or removed panels or fixtures, and more construction materials or debris on the floor. The overall impression is one of a space in the midst of transformation or repair."
    # x_1 = "A series of videos showing a room undergoing renovation or construction, with various stages of progress and tools and materials scattered around"
    x_1 = "A series of videos capturing a person in various stages of home improvement or construction projects, with different perspectives and environments." # 
    x_2 = "A series of videos showcasing outdoor work and DIY projects, including construction, maintenance, and repair tasks with various tools and equipment."
    x_3 = "A person in a home improvement project, kneeling on the floor with tools and equipment around them."
    x_4 = "The board seemed to be the wrong size and I needed to reconstructe it myself using a power cutter. Can you tell me where I can find my power cutter?"

    # 2. encode
    inputs_1 = tokenizer(x_1, padding=True, return_tensors='pt')
    for k, v in inputs_1.items():
        inputs_1[k] = v.cuda()
    outputs_1 = model(**inputs_1).last_hidden_state
    embeddings_1 = pooling(outputs_1, inputs_1, 'cls')
    
    inputs_2 = tokenizer(x_2, padding=True, return_tensors='pt')
    for k, v in inputs_2.items():
        inputs_2[k] = v.cuda()
    outputs_2 = model(**inputs_2).last_hidden_state
    embeddings_2 = pooling(outputs_2, inputs_2, 'cls')
    
    inputs_3 = tokenizer(x_3, padding=True, return_tensors='pt')
    for k, v in inputs_3.items():
        inputs_3[k] = v.cuda()
    outputs_3 = model(**inputs_3).last_hidden_state
    embeddings_3 = pooling(outputs_3, inputs_3, 'cls')
        
    inputs_4 = tokenizer(x_4, padding=True, return_tensors='pt')
    for k, v in inputs_4.items():
        inputs_4[k] = v.cuda()
    outputs_4 = model(**inputs_4).last_hidden_state
    embeddings_4 = pooling(outputs_4, inputs_4, 'cls')
    
    print("{}/{}".format(embeddings_1.shape, embeddings_2.shape))
    # embeddings = pooling(outputs, inputs, 'mean')
    # print(outputs[0].shape)
    similarities1 = cos_sim(embeddings_4[0], embeddings_1[0])
    similarities2 = cos_sim(embeddings_4[0], embeddings_2[0])
    similarities3 = cos_sim(embeddings_4[0], embeddings_3[0])
    print('similarities:{}/{}/{}'.format(similarities1, similarities2, similarities3))


def simulate_memory_construct_ori():
    from collections import deque, defaultdict
    
    shape = (1, 144, 4096)
    
    time_line = [812, 1189, 3161, 4147, 7975, 10440]
    
    # real_time = threading 
    buffer_size = [414, 673, 819, 943, 1486, 1667]
    buffer = []
    chunk_size = 40
    num_clusters = 5
    interval = 3
    start_update = False
    all_nodes = deque([])
    buffer_lock = threading.Lock()
    update_event = threading.Event()
    finish_event = threading.Event()
    stop_event = threading.Event()
    
    def print_tree(node, indent=0):
        """打印树状结构"""
        if isinstance(node, list):
            for single_node in node:
                print(" " * indent + f"{RED}Depth{RESET}: {single_node.depth}, {RED}Centroids{RESET}: {single_node.centroids.shape}, {RED}Text{RESET} :{single_node.text}")
                if single_node.children is not None:
                    for child in single_node.children:
                        print_tree(child, indent + 4)
                
        else:
            print(" " * indent + f"{RED}Depth{RESET}: {node.depth}, {RED}Centroids{RESET}: {node.centroids.shape}, {RED}Text{RESET} :{node.text}")
            for child in node.children:
                print_tree(child, indent + 4)
    
    def count_nodes_by_depth(nodes):
        depth_count = defaultdict(int)
        for node in nodes:
            depth_count[node.depth] += 1
            stack = node.children.copy()
            while stack:
                current_node = stack.pop()
                depth_count[current_node.depth] += 1
                stack.extend(current_node.children)
        return depth_count
    
    def get_summarize_depth(nodes, interval):
        depth_count = defaultdict(int)
        depth_count.clear()
        for node in nodes:
            depth_count[node.depth] += 1
            # stack = node.children.copy()
            # while stack:
            #     current_node = stack.pop()
            #     depth_count[current_node.depth] += 1
            #     stack.extend(current_node.children)
                
        # 找出最高优先级的深度
        max_depth = max(depth_count.keys())
        for depth in range(max_depth, -1, -1):
            if depth_count[depth] % interval == 0 and depth_count[depth] > 0: # 判断目前需要使用的深度
                return depth, depth_count
        return 0 ,depth_count
        
        
    
    def simulate_building_memory_tree(k_means_chunk_feature_list, num_clusters, interval, chunked_feature_list, existing_tree=None, past_feature_length=0):
        output_list = []  # prepare summary
        for chunk_feature in chunked_feature_list:
            dimension = chunk_feature[0].shape[-1]
            chunk_feature = torch.cat(chunk_feature, dim=0).reshape(-1, dimension)
            time.sleep(2)  # 模拟计算时间
            output_list.append("test only hahahahaha")

        # past_feature_length += 
        
        nodes = [MultimodalTreeNode(tensor, text, depth=0) for (tensor, text) in zip(k_means_chunk_feature_list, output_list)]
        
        if existing_tree:
            # nodes.append(existing_tree)
            nodes = existing_tree + nodes
        
        summarize_depth, depth_count = get_summarize_depth(nodes, interval)
        
        start_index = next((index for index, node in enumerate(nodes) if node.depth == summarize_depth), None)
        chunk_length = len([x for x in nodes if x.depth == summarize_depth])
        print("summarize_depth:{}/ start_index:{}/ chunk_length:{} / len(nodes):{}".format(summarize_depth, start_index, chunk_length, len(nodes)))
        
        if chunk_length % interval == 0 and len(nodes) > 0: # for first
            
            
            print("building summarize node for {} clip".format(start_index + 1))
            
            
            # for i in range(0 + clip * interval, len(nodes), interval):
            # chunk = nodes[(clip - 1) * interval : (clip - 1) * interval + interval]
            chunk = nodes[start_index: start_index + interval]
            centroids_list = [node.centroids for node in chunk]
            # caption_list = [node.text for node in chunk]
            combined_centroids = torch.cat(centroids_list, dim=0)

            if combined_centroids.shape[0] > num_clusters:
                new_centroids, labels = weighted_kmeans_feature(combined_centroids, num_clusters)
            else:
                new_centroids = combined_centroids

            summarize_text = 'OKOKOKOKOKO'
            time.sleep(2)  # 模拟计算时间
            new_node = MultimodalTreeNode(new_centroids, summarize_text, depth=chunk[0].depth + 1)

            for j, node in enumerate(chunk):
                new_node.children.append(node) # 将之前的list全部进行总结
            
            nodes[start_index: start_index + interval] = [new_node]
            
            return nodes
        
        else:
            return nodes
        
    
    def simulate_video_reader():
        nonlocal start_update
        for time_spot in tqdm.tqdm(range(buffer_size[-1]), desc="time"):
            buffer.append(torch.randn(shape).cuda())
            if len(buffer) >= chunk_size and len(buffer) % chunk_size == 0:
                start_update = True
                update_event.set()  # 通知更新线程
                # update_event.clear()  # 清除事件，以便更新线程等待处理完毕后再继续
            
            if start_update:
                finish_event.wait()  # 等待更新线程处理完毕后再继续
                
            if stop_event.is_set():
                break
            time.sleep(0.0005)
    
    def simulate_update_buffer():
        nonlocal start_update, buffer
        existing_tree = None
        # buffer_cahce = None
        while not stop_event.is_set():
            update_event.wait()  # 等待视频读取线程通知更新
            finish_event.clear()
            if start_update:
                # if buffer_cahce is None:
                #     buffer_cahce = buffer
                # else:
                # buffer_cahce = buffer
                if existing_tree is not None:
                    print("last length:{}".format(length))
                    print("buffer length:{}".format(len(buffer)))
                    buffer_cahce = buffer[length:] # out of index 
                    print("buffer_cahce length:{}".format(len(buffer_cahce)))
                    length += len(buffer_cahce) 
                    
                else:
                    buffer_cahce = buffer
                    print("buffer_cahce length:{}".format(len(buffer_cahce)))
                    length = len(buffer_cahce)
                    
                time_1 = time.time()
                chunk_feature_list = [buffer_cahce[i:i + chunk_size] for i in range(0, len(buffer_cahce), chunk_size)]
                k_means_chunk_feature_list = [
                    weighted_kmeans_feature(torch.cat(chunk_feature), num_clusters)[0] if len(chunk_feature) >= chunk_size else torch.cat(chunk_feature)
                    for chunk_feature in chunk_feature_list
                ]
                print("<<<<<<< updating long memory tree >>>>>>>>>>>")
                
                existing_tree = simulate_building_memory_tree(
                    k_means_chunk_feature_list, 
                    num_clusters, 
                    interval, 
                    chunk_feature_list, 
                    existing_tree)
                # buffer = buffer[len(buffer) % chunk_size:]  # 保留未处理的buffer数据
                start_update = False
                time_2 = time.time()
                print_tree(existing_tree)
                print("spend_time:{}".format(time_2 - time_1))
                print("<<<<<<< updating long memory finish >>>>>>>>>>>")
                depth_count = count_nodes_by_depth(existing_tree)

                print("节点深度统计:")
                for depth, count in depth_count.items():
                    print(f"{BLUE}深度 {depth}{RESET}: {count} 个节点")
            else:
                time.sleep(0.001)
            finish_event.set()  # 通知视频读取线程继续
        time.sleep(0.001)

        
    video_thread = threading.Thread(target=simulate_video_reader)
    
    update_thread = threading.Thread(target=simulate_update_buffer)
    
    video_thread.start()
    update_thread.start()

    video_thread.join()
    stop_event.set()  # 确保更新线程也能正确退出
    update_thread.join()
    
    
def simulate_memory_construct():
    from collections import deque, defaultdict
    
    shape = (1, 144, 4096)
    
    time_line = [812, 1189, 3161, 4147, 7975, 10440]
    
    # real_time = threading 
    buffer_size = [414, 673, 819, 943, 1486, 1667]
    buffer = []
    chunk_size = 40
    num_clusters = 5
    interval = 10
    processed_length = 0
    buffer_cache = None
    start_update = False
    existing_tree = None
    update_finish = True
    all_nodes = deque([])
    buffer_lock = threading.Lock()
    update_event = threading.Event()
    finish_event = threading.Event()
    stop_event = threading.Event()
    
    print("Ready models")
    embedding_model_id = '/13390024681/All_Model_Zoo/mxbai-colbert-large-v1'
    embedding_tokenizer = AutoTokenizer.from_pretrained(embedding_model_id)
    embedding_model = AutoModel.from_pretrained(embedding_model_id).cuda()
    print("model readed ")
        
    def print_tree(node, indent=0):
        """打印树状结构"""
        if isinstance(node, list):
            for single_node in node:
                print(" " * indent + f"{RED}Depth{RESET}: {single_node.depth}, {RED}Centroids{RESET}: {single_node.centroids.shape}, {RED}Text{RESET} :{single_node.text}")
                if single_node.children is not None:
                    for child in single_node.children:
                        print_tree(child, indent + 4)
                
        else:
            print(" " * indent + f"{RED}Depth{RESET}: {node.depth}, {RED}Centroids{RESET}: {node.centroids.shape}, {RED}Text{RESET} :{node.text}")
            for child in node.children:
                print_tree(child, indent + 4)
    
        depth_count = defaultdict(int)
        for node in nodes:
            depth_count[node.depth] += 1
            stack = node.children.copy()
            while stack:
                current_node = stack.pop()
                depth_count[current_node.depth] += 1
                stack.extend(current_node.children)
        return depth_count
    
    def get_summarize_depth(nodes, interval):
        depth_count = defaultdict(int)
        depth_count.clear()
        for node in nodes:
            depth_count[node.depth] += 1
                
        # 找出最高优先级的深度
        max_depth = max(depth_count.keys())
        for depth in range(max_depth, -1, -1):
            if depth_count[depth] % interval == 0 and depth_count[depth] > 0: # 判断目前需要使用的深度
                return depth, depth_count
        return 0 ,depth_count
    
    def search_tree_multi_modal_with_each_node(all_nodes, query, image_embedding, model, tokenizer, top_k=1):
        """在树中沿着每一层相似度最高的分支不断寻找，并将每一层对应的特征都提取出来"""
        # The model works really well with cls pooling (default) but also with mean pooling.
        def pooling(outputs: torch.Tensor, inputs,  strategy: str = 'cls') -> np.ndarray:
            if strategy == 'cls':
                outputs = outputs[:, 0]
            elif strategy == 'mean':
                outputs = torch.sum(
                    outputs * inputs["attention_mask"][:, :, None], dim=1) / torch.sum(inputs["attention_mask"])
            else:
                raise NotImplementedError
            return outputs.detach().cpu().numpy()
        
        path_features = []
        path_text = []
        redundant_nodes = []
        # x = torch.nn.functional.normalize(torch.cat([query, image_embedding], dim=0), p=2, dim=0)
        print("Question:{}".format(query))
        query_ids = tokenizer(query, padding=True, return_tensors='pt')
        for k, v in query_ids.items():
            query_ids[k] = v.cuda()
        outputs_1 = model(**query_ids).last_hidden_state
        query_embedding = pooling(outputs_1, query_ids, 'cls')
        
        for node in all_nodes:
            current_node = node
            if current_node.depth == 0:
                redundant_nodes.append(current_node)                
            else:
                while current_node.children :
                    # if current_node.
                    best_child_index = None
                    best_sim = 0
                    
                    # 计算查询张量与每个子节点的所有特征张量之间的相似度
                    for i, child in enumerate(current_node.children):
                        # distances = torch.cdist(query.unsqueeze(0), child.centroids.view(-1, query.size(-1)).unsqueeze(0)).squeeze(0)
                        # text_tensor = torch.tensor(tokenizer(current_node.text).input_ids , dtype=torch.float16).unsqueeze(0).cuda()
                        print("text:{}".format(child.text))
                        text_ids = tokenizer(child.text, padding=True, return_tensors='pt')
                        for k, v in text_ids.items():
                            text_ids[k] = v.cuda()
                        outputs_2 = model(**text_ids).last_hidden_state
                        text_embedding = pooling(outputs_2, text_ids, 'cls')
                        # distance_text  = (x @ torch.nn.functional.normalize(text_embeddings, p=2, dim=0).permute(1, 0)).mean(0).mean(0) # remember to normalize the embedding
                        # distance_image = (x @ torch.nn.functional.normalize(child.centroids.view(-1, query.size(-1)), p=2, dim=0).permute(1, 0)).mean(0).mean(0)
                        sim = cos_sim(text_embedding[0], query_embedding[0])
                        # current_node.text_distance = distance_text
                        # current_node.image_distance = distance_image
                        print(f"{RED}sim{RESET}:{sim} in depth:{current_node.depth} {i} child")
                        # distance = (query @ child.centroids.view(-1, query.size(-1)).permute(1, 0)).sum(0).sum(0) # len x 1024 @ len x 1024 = len x len
                        # min_distance, _ = distances.min(dim=0)
                        if sim > best_sim:
                            best_sim = sim
                            best_child_index = i
                        else:
                            continue
                        
                    print(f"{BLUE}best_child_index:{RESET}{best_child_index}")
                    # 选择相似度最高的子节点
                    path_features.append(current_node.children[best_child_index].centroids)
                    path_text.append(current_node.children[best_child_index].text)
                    current_node = current_node.children[best_child_index]
        
        # 针对存在的冗余节点再进行一次计算
        best_index = 0
        best_sim = 0
        
        if len(redundant_nodes) >= 1:
            for  i,node in enumerate(redundant_nodes):
                print("text:{}".format(node.text))
                text_ids = tokenizer(node.text, padding=True, return_tensors='pt')
                for k, v in text_ids.items():
                    text_ids[k] = v.cuda()
                outputs_2 = model(**text_ids).last_hidden_state
                text_embedding = pooling(outputs_2, text_ids, 'cls')
                # distance_text  = (x @ torch.nn.functional.normalize(text_embeddings, p=2, dim=0).permute(1, 0)).mean(0).mean(0) # remember to normalize the embedding
                # distance_image = (x @ torch.nn.functional.normalize(child.centroids.view(-1, query.size(-1)), p=2, dim=0).permute(1, 0)).mean(0).mean(0)
                sim = cos_sim(text_embedding[0], query_embedding[0])
                # current_node.text_distance = distance_text
                # current_node.image_distance = distance_image
                print(f"{RED}sim{RESET}:{sim} in depth:{node.depth} {i} child")
                # distance = (query @ child.centroids.view(-1, query.size(-1)).permute(1, 0)).sum(0).sum(0) # len x 1024 @ len x 1024 = len x len
                # min_distance, _ = distances.min(dim=0)
                if sim > best_sim:
                    best_sim = sim
                    best_index = i
                else:
                    continue
                
            print(f"{BLUE}best_redundant_index:{RESET}{best_index}")
            path_features.append(redundant_nodes[best_index].centroids)
            path_text.append(redundant_nodes[best_index].text)
        # else:
        #     # using the nearst feature 
        #     best_index = 0
            
        
        # path_features.append(current_node.centroids)  # 添加最底层节点的特征
        # path_text.append(current_node.text)

        return path_features, path_text
        
    
    def simulate_building_memory_tree(k_means_chunk_feature_list, num_clusters, interval, chunked_feature_list, existing_tree=None, past_feature_length=0):
        output_list = []  # prepare summary
        for chunk_feature in chunked_feature_list:
            dimension = chunk_feature[0].shape[-1]
            chunk_feature = torch.cat(chunk_feature, dim=0).reshape(-1, dimension)
            time.sleep(1)  # 模拟计算时间
            output_list.append("test only hahahahaha")

        # past_feature_length += 
        
        nodes = [MultimodalTreeNode(tensor, text, depth=0) for (tensor, text) in zip(k_means_chunk_feature_list, output_list)]
        
        if existing_tree:
            # nodes.append(existing_tree)
            nodes = existing_tree + nodes
        
        summarize_depth, depth_count = get_summarize_depth(nodes, interval)
        
        start_index = next((index for index, node in enumerate(nodes) if node.depth == summarize_depth), None)
        chunk_length = len([x for x in nodes if x.depth == summarize_depth])
        print("summarize_depth:{}/ start_index:{}/ chunk_length:{} / len(nodes):{}".format(summarize_depth, start_index, chunk_length, len(nodes)))
        
        if chunk_length % interval >= 0 and len(nodes) > 0 and chunk_length >= interval: # for first
                        
            print("building summarize node for {} clip".format(start_index + 1))
            
            
            # for i in range(0 + clip * interval, len(nodes), interval):
            # chunk = nodes[(clip - 1) * interval : (clip - 1) * interval + interval]
            chunk = nodes[start_index: start_index + interval]
            centroids_list = [node.centroids for node in chunk]
            # caption_list = [node.text for node in chunk]
            combined_centroids = torch.cat(centroids_list, dim=0)

            if combined_centroids.shape[0] > num_clusters:
                new_centroids, labels = weighted_kmeans_feature(combined_centroids, num_clusters)
            else:
                new_centroids = combined_centroids

            summarize_text = 'OKOKOKOKOKO'
            time.sleep(1)  # 模拟计算时间
            new_node = MultimodalTreeNode(new_centroids, summarize_text, depth=chunk[0].depth + 1)

            for j, node in enumerate(chunk):
                new_node.children.append(node) # 将之前的list全部进行总结
            
            nodes[start_index: start_index + interval] = [new_node]
            
            return nodes
        
        else:
            return nodes
    
    # def simulate_building_memory_tree(k_means_chunk_feature_list, num_clusters, interval, chunked_feature_list, existing_tree=None):
    #     output_list = []  # prepare summary
    #     for chunk_feature in chunked_feature_list:
    #         dimension = chunk_feature[0].shape[-1]
    #         chunk_feature = torch.cat(chunk_feature, dim=0).reshape(-1, dimension)
    #         time.sleep(2)  # simulate processing time
    #         output_list.append("test only hahahahaha")

    #     nodes = [MultimodalTreeNode(tensor, text, depth=0) for (tensor, text) in zip(k_means_chunk_feature_list, output_list)]
        
    #     if existing_tree:
    #         nodes = existing_tree + nodes

    #     while True:
    #         summarize_depth, depth_count = get_summarize_depth(nodes, interval)
            
    #         start_index = next((index for index, node in enumerate(nodes) if node.depth == summarize_depth), None)
    #         chunk_length = len([x for x in nodes if x.depth == summarize_depth])
            
    #         if summarize_depth == 0 and start_index is None:
    #             break
            
    #         if chunk_length % interval == 0 and len(nodes) > 0:
    #             chunk = nodes[start_index: start_index + interval]
    #             centroids_list = [node.centroids for node in chunk]
    #             combined_centroids = torch.cat(centroids_list, dim=0)

    #             if combined_centroids.shape[0] > num_clusters:
    #                 new_centroids, labels = weighted_kmeans_feature(combined_centroids, num_clusters)
    #             else:
    #                 new_centroids = combined_centroids

    #             summarize_text = 'OKOKOKOKOKO'
    #             time.sleep(2)  # simulate processing time
    #             new_node = MultimodalTreeNode(new_centroids, summarize_text, depth=chunk[0].depth + 1)

    #             for node in chunk:
    #                 new_node.children.append(node)
                
    #             nodes[start_index: start_index + interval] = [new_node]
    #         else:
    #             break

    #     return nodes

    
    def simulate_video_reader():
        nonlocal start_update, buffer_cache, existing_tree, buffer, update_finish
        for time_spot in tqdm.tqdm(range(buffer_size[-1]), desc="time"):
            buffer.append(torch.randn(shape).cuda())
            # print("update_finish:{}".format(update_finish))
            if len(buffer) >= chunk_size and len(buffer) % chunk_size == 0 and update_finish:
                print("load vision buffer")
                # buffer_cache = buffer
                if existing_tree is not None:
                    # print("last length:{}".format(length))
                    # print("buffer length:{}".format(len(buffer)))
                    buffer_cache = buffer[length:] # out of index 
                    # print("buffer_cahce length:{}".format(len(buffer_cahce)))
                    length += len(buffer_cache) 
                    
                else:
                    buffer_cache = buffer
                    # print("buffer_cahce length:{}".format(len(buffer_cahce)))
                    length = len(buffer_cache)
                print("vision buffer loaded ")
                print(len(buffer_cache))
                processed_length = length 
                # if processed_length < len(buffer):
                start_update = True
                print("update_finish",start_update)
                # else:
                    # start_update = False
                
            time.sleep(0.05)
    
    def simulate_update_buffer():
        nonlocal start_update, buffer_cache, existing_tree,update_finish
        
        while True:

            if start_update :
                update_finish = False
                time_1 = time.time()
                chunk_feature_list = [buffer_cache[i:i + chunk_size] for i in range(0, len(buffer_cache), chunk_size)]
                k_means_chunk_feature_list = [
                    weighted_kmeans_feature(torch.cat(chunk_feature), num_clusters)[0] if len(chunk_feature) >= chunk_size else torch.cat(chunk_feature)
                    for chunk_feature in chunk_feature_list
                ]
                # print("<<<<<<< updating long memory tree >>>>>>>>>>>")
                
                existing_tree = simulate_building_memory_tree(
                    k_means_chunk_feature_list, 
                    num_clusters, 
                    interval, 
                    chunk_feature_list, 
                    existing_tree)
                # buffer = buffer[len(buffer) % chunk_size:]  # 保留未处理的buffer数据
                start_update = False
                update_finish = True
                time_2 = time.time()
                print_tree(existing_tree)
                print("spend_time:{}".format(time_2 - time_1))
                # print("<<<<<<< updating long memory finish >>>>>>>>>>>")
                depth_count = count_nodes_by_depth(existing_tree)

                print("节点深度统计:")
                for depth, count in depth_count.items():
                    print(f"{BLUE}深度 {depth}{RESET}: {count} 个节点")

                # finish_event.set()  # 通知视频读取线程继续
                
            else:
                time.sleep(0.001)

    def simulate_question():
        nonlocal start_update, buffer_cache, existing_tree, buffer
        start_inference = False
        while True:
            if len(buffer) in buffer_size:
                start_inference = True
                
            if start_inference:
                question_text = "I need you now , please tell me what you see from these imag"
                if start_update:
                    # print("Waiting Tree build finished ")
                    time.sleep(0.01)
                else:
                    tree_cache = existing_tree
                    path_features, path_text = search_tree_multi_modal_with_each_node(tree_cache, question_text, None, embedding_model, embedding_tokenizer)
                    print("path feature:{}".format(len(path_features)))
                    start_inference = False
                    
            else:
                time.sleep(0.04)
        
    video_thread = threading.Thread(target=simulate_video_reader)
    update_thread = threading.Thread(target=simulate_update_buffer)
    qa_thread = threading.Thread(target=simulate_question)
    
    video_thread.start()
    update_thread.start()
    qa_thread.start()

    video_thread.join()
    stop_event.set()  # 确保更新线程也能正确退出
    update_thread.join()
    qa_thread.join()
            

if __name__ == "__main__":
    # test_memory()
    # test_eval_metrics()
    # test_embedding()
    simulate_memory_construct()

    # # 打印初始显存使用情况
    # print(f"Initial allocated memory: {torch.cuda.memory_allocated()/ (1024 ** 3)} GB")
    # print(f"Initial reserved memory: {torch.cuda.memory_reserved()/ (1024 ** 3)} GB")
    
    # time_1 = time.time()
    # # x = torch.randn((10000, 576, 1024)).cuda()   # (BS, N, D)# 定义张量的形状
    # shape = (1, 576, 4096)
    
    # # 构建包含10000个元素的列表，每个元素是形状为[576, 1024]的张量 分开进行计算
    # tensor_list = [torch.randn(shape).cuda() for _ in range(582)]
    # # tensor_list  = list(torch.split(torch.randn(shape).cuda(), 1))
    # time_2 = time.time()
    # print("tensor list created !!")
    # # 使用原始方法进行直接构建
    # # tensor_list = compress_spatial_features(tensor_list, 2)
    # # reduced_feature, labels = weighted_kmeans_feature(torch.cat(tensor_list), 10)
    # # 构建树状结构
    # # tree = building_memory_tree(tensor_list, 2, 100, 10, 0)
    # # tree = test_2(tensor_list)
    # short_memory_buffer, long_memory_tree = test_3(tensor_list) # 构建完整的树状结构
    # print("memory tree building finish")
    # time_3 = time.time()
    
    # def print_tree(node, indent=0):
    #     """打印树状结构"""
    #     print(" " * indent + f"Depth: {node.depth}, Centroids Shape: {node.centroids.shape}")
    #     for child in node.children:
    #         print_tree(child, indent + 4)
            
    # print_tree(long_memory_tree)
    
    # # 创建一个查询张量
    # query = torch.randn((13, 4096)).to(tensor_list[0].device)

    # # 在树中搜索与查询最相似的特征
    # closest_features = search_tree(long_memory_tree, query)
    # print("tree search finish !!")
    
    # visualize_memory_feature_with_PCA(tensor_list, closest_features, clustering=5, same_color=False, only_most_important=True)

    # time_4 = time.time()
    
    # # # 打印最相似特征的形状
    # # for feature in closest_features:
    # #     print(f"Closest feature shape: {feature.shape}")
    # print("short memory buffer:{}/{}".format(len(short_memory_buffer), short_memory_buffer[0].shape))
    # print("long memory buffer:{}/{}".format(len(closest_features), closest_features[0].shape))
    # print(f"Time elapsed: {time_3 - time_2}/{time_2 - time_1} seconds")
    # # 打印最终显存使用情况
    # print(f"Final allocated memory: {torch.cuda.memory_allocated()/ (1024 ** 3)} GB")
    # print(f"Final reserved memory: {torch.cuda.memory_reserved()/ (1024 ** 3)} GB")
    