import os
import json
import json_lines
import numpy as np

data_path = '/13390024681/llama/EfficientVideo/All_Score/Ego_Streaming/LongVA/merge.jsonl'
# 分组数
group_size = 6

# 初始化变量存储每个位置的总分和计数
sum_scores = [0] * group_size
count_scores = [0] * group_size

# 初始化变量存储每个组的正确预测计数和总预测计数
correct_predictions = [0] * group_size
total_predictions = [0] * group_size

with open(data_path, "rb") as f:
    data = json_lines.reader(f)
    # 计算每个位置的总分和计数，并计算准确率
    for i, entry in enumerate(data):
        position = i % group_size
        sum_scores[position] += entry['score']
        count_scores[position] += 1
        total_predictions[position] += 1
        if entry['llama_pred'] == 'yes':
            correct_predictions[position] += 1

# 计算平均分
average_scores = [sum_scores[i] / count_scores[i] for i in range(group_size)]

# 计算准确率
accuracies = [correct_predictions[i] / total_predictions[i] for i in range(group_size)]

avg_score = np.mean(average_scores)
avg_acc = np.mean(accuracies)


print("各位置的平均分数：", average_scores, avg_score)
print("各位置的准确率：", accuracies, avg_acc)