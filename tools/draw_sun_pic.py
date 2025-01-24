import matplotlib.pyplot as plt

# 定义数据
data = {
    'ScienceQA': {
        'Geography': 16.68,
        'Physics': 12.61,
        'Biology': 12.02,
        'Earth-Science': 5.05,
        'Chemistry': 3.09,
        'Economics': 2.84,
        'History': 3.17,
        'Practices': 3.20
    },
    'MMMU': {
        'Medicine': 4.91,
        'Art': 3.27,
        'Science': 2.71,
        'Business': 3.65
    },
    'MathVista': {
        'GPS': 9.49
    },
    'TQA': 5.00,
    'FQA': 5.43,
    'VQA': 4.30
}

# 绘制饼图的函数
def plot_pie(ax, sizes, labels, colors):
    wedges, texts, autotexts = ax.pie(
        sizes, labels=labels, colors=colors, autopct='%1.2f%%', startangle=90,
        wedgeprops=dict(width=0.3, edgecolor='w'))

    # 设置字体大小
    for text in texts + autotexts:
        text.set_fontsize(10)

# 创建绘图，增加分辨率和尺寸
fig, ax = plt.subplots(figsize=(10, 10), dpi=150)

# 定义颜色
colors = plt.get_cmap('tab20c')(range(len(data)))

# 绘制外层饼图
outer_sizes = [sum(data[group].values()) if isinstance(data[group], dict) else data[group] for group in data]
outer_labels = list(data.keys())
plot_pie(ax, outer_sizes, outer_labels, colors)

# 绘制内层饼图
inner_sizes = [size for group in data.values() if isinstance(group, dict) for size in group.values()]
inner_labels = [label for group in data.values() if isinstance(group, dict) for label in group.keys()]
inner_colors = plt.get_cmap('tab20b')(range(len(inner_labels)))
ax.pie(inner_sizes, labels=inner_labels, colors=inner_colors, radius=0.7, startangle=90,
       wedgeprops=dict(width=0.3, edgecolor='w'))

# 确保绘图为圆形
ax.set(aspect="equal")

plt.show()
plt.savefig("./sun.jpg")