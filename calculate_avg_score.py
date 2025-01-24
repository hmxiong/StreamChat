import json
from collections import defaultdict

# 假设你的 JSONL 文件名为 data.jsonl
jsonl_file_path = '/13390024681/llama/EfficientVideo/All_Score/GPT_4o/streamingbench_merge.jsonl'

# 初始化统计变量
class_scores = defaultdict(int)
class_counts = defaultdict(int)
class_acc_counts = defaultdict(int)  # 用于统计每个类别的样本数

score_differences = []  # 用于存储相邻数据的 score 差值
process_time = []
previous_score = None  # 保存前一个数据的 score

# 读取 JSONL 文件并统计每个 class 的 score 和 acc
with open(jsonl_file_path, 'r') as file:
    for line in file:
        data = json.loads(line)
        class_type = data['class']
        score = data['score']
        acc = 1 if data['llama_pred'] == 'yes' else 0
        
        class_scores[class_type] += score
        class_counts[class_type] += acc
        class_acc_counts[class_type] += 1

        # process_time.append(data['process_time'])
        # 计算相邻两个 score 的差值
        if previous_score is not None:
            score_difference = abs(score - previous_score)
            score_differences.append(score_difference)
        
        previous_score = score  # 更新 previous_score 为当前 score

# 计算平均差值
if score_differences:
    average_difference = sum(score_differences) / len(score_differences)
else:
    average_difference = 0

# avg_process_time = sum(process_time) / len(process_time)
# 输出统计结果
print("每个类别的总得分 (score) 和准确性 (acc)，以及平均分:")
for class_type in class_scores.keys():
    total_score = class_scores[class_type]
    total_acc = class_counts[class_type]
    count = class_acc_counts[class_type]
    avg_score = total_score / count
    avg_acc = total_acc / count
    print(f"类别 {class_type}: 总得分 {total_score}, 总准确性 {total_acc}, 平均得分 {avg_score:.2f}, 平均准确性 {avg_acc:.4f}")

# print("avg process time:{}".format(avg_process_time))
# 输出平均差值
print(f"相邻数据的 score 平均差值: {average_difference:.2f}")