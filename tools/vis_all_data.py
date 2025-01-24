import plotly.graph_objects as go
import plotly.io as pio

# 数据
# data = {
#     "Ego":{"Cooking":305, "Construction":395, "Room-Tour":320, "Gardening":101, "Others":379, "All":1500},
#     "Movie":{"Drama":155, "Action":153, "Sci-fi":24, "Romance":11, "Cartoon":18, "All":361},
#     "Sport":{"Basketball moves": 459, "Football boot":482, "Acrobatic gymnastics": 479, "Beach volleyball": 250, "All":1670},
#     "WebVideo":{"Comedy (drama)": 405, "Talent show": 456, "Apple TV": 208, "Cooking show": 456, "All":1526}
# }
# data = {
#     "Ego":{"Cooking":13, "Construction":14, "Room-Tour":24, "Gardening":1, "Others":8, "All":60},
#     # "Movie":{"Drama":155, "Action":153, "Sci-fi":24, "Romance":11, "Cartoon":18, "All":361},
#     "WorkingVideo":{"Cook":25, "Metalworking":19, "All":44},
#     "WebVideo":{"Comedy (drama)": 14, "Apple TV": 1, "Cooking show": 36, "All":51}
# }

data = {
    "Ego":{"Cooking":20, "Construction":24, "Room-Tour":43, "Others":13, "All":100},
    # "Movie":{"Drama":155, "Action":153, "Sci-fi":24, "Romance":11, "Cartoon":18, "All":361},
    "Working":{"Cook":25, "Metalworking":19, "Plant":15, "All":59},
    "WebVideo":{"Comedy (drama)": 14, "Show": 36, "Bathroom":18, "Classroom":17, "Livingroom":14, "Outdoor":13, "All":112},
    "Movie":{"Action": 16, "Sci-fi": 5, "Romance":13, "All":34}
}

# 目前的视频总量 1500 + 361 + 1670 + 1526 = 5057 按照500个视频来算的话平均每天需要打24个视频才能保证一周的时间能够完成

# 准备数据
labels = []
parents = []
values = []

for main_category, subcategories in data.items():
    for subcategory, count in subcategories.items():
        if subcategory != "All":
            labels.append(subcategory)
            parents.append(main_category)
            values.append(count)
    labels.append(main_category)
    parents.append("")
    values.append(subcategories["All"])

# 创建Sunburst图
fig = go.Figure(go.Sunburst(
    labels=labels,
    parents=parents,
    values=values,
    branchvalues="total",
    hoverinfo="label+value+percent entry",
    textinfo="label+percent entry",
    textfont=dict(size=25, color='white')  # 设置内部文字的字体大小
))

# 更新布局
fig.update_layout(
    margin=dict(t=0, l=0, r=0, b=0),
    uniformtext=dict(minsize=10, mode='hide'),  # 调整文本大小，避免重叠
    title_font_size=20,  # 设置标题字体大小
    font=dict(size=25, color='white')  # 设置整体字体大小
)

# 保存图表为PNG图片
fig.write_image("/13390024681/llama/EfficientVideo/Ours/tools/Ego_Class/streaming_bench_v0.3.png")

# 展示图表
fig.show()