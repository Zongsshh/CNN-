import os

# 定义要创建的文件夹结构
folders = [
    'data',
    'models',
    'results'
]

# 创建文件夹
for folder in folders:
    if not os.path.exists(folder):
        os.makedirs(folder)
        print(f"✓ 创建文件夹: {folder}")
    else:
        print(f"○ 文件夹已存在: {folder}")

# 创建空的Python文件
files = {
    'main.py': '# 手写数字识别主程序',
    'utils.py': '# 工具函数',
    'models/model.py': '# 模型定义',
    'README.md': '# 手写数字识别实验'
}

for filename, content in files.items():
    # 确保文件所在目录存在
    directory = os.path.dirname(filename)
    if directory and not os.path.exists(directory):
        os.makedirs(directory)

    # 创建文件
    if not os.path.exists(filename):
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"✓ 创建文件: {filename}")
    else:
        print(f"○ 文件已存在: {filename}")

print("\n✅ 文件夹结构创建完成！")
print("\n项目结构：")
print("""
your_project/
├── data/              # 存放数据集
├── models/
│   └── model.py       # 模型定义
├── results/           # 存放训练结果
├── main.py           # 主程序
├── utils.py          # 工具函数
└── README.md         # 项目说明
""")