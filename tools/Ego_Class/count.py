import json

# 初始化一个空的字典来存储每个类别的数据数量
category_count = {}

# 读取 JSONL 文件
file_path = '/13390024681/llama/EfficientVideo/Ours/tools/Ego_Class/merge.jsonl'

with open(file_path, 'r') as file:
    for line in file:
        # 解析 JSON 行
        data = json.loads(line.strip())
        category = data.get('category')
        
        # 更新类别计数
        if category in category_count:
            category_count[category] += 1
        else:
            category_count[category] = 1

# 打印每个类别的数据数量
for category, count in category_count.items():
    print(f"Category: {category}, Count: {count}")

# 将结果保存到一个 JSON 文件
output_file = 'category_counts.json'
with open(output_file, 'w') as outfile:
    json.dump(category_count, outfile, indent=4)

print(f"Category counts saved to {output_file}")