import json
from collections import defaultdict

# 读取 JSON 文件
file_path = '/13390024681/llama/EfficientVideo/Ours/tools/Youtube_movie.json'
with open(file_path, 'r') as file:
    data = json.load(file)

# 统计 category 数量
category_counts = defaultdict(int)
for item in data:
    category = item.get('category')
    if category:
        category_counts[category] += 1

# 打印 category 统计结果
for category, count in category_counts.items():
    print(f"Category: {category}, Count: {count}")