import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

# 设置随机种子，保证结果可重复
torch.manual_seed(42)

# 定义数据预处理：将图片转换为张量，并归一化到[0,1]范围
transform = transforms.Compose([
    transforms.ToTensor(),
])

# 下载并加载训练数据集
print("📥 正在下载MNIST训练数据集...")
train_dataset = datasets.MNIST(
    root='./data',      # 数据保存路径
    train=True,         # 加载训练集
    download=True,      # 如果没有就下载
    transform=transform # 应用预处理
)

# 下载并加载测试数据集
print("📥 正在下载MNIST测试数据集...")
test_dataset = datasets.MNIST(
    root='./data',
    train=False,        # 加载测试集
    download=True,
    transform=transform
)

print(f"\n✅ 训练集加载完成！共有 {len(train_dataset)} 张图片")
print(f"✅ 测试集加载完成！共有 {len(test_dataset)} 张图片")

# 查看第一张图片
print("\n🔍 查看第一张训练图片：")
first_image, first_label = train_dataset[0]
print(f"图片形状: {first_image.shape}")  # 应该是 [1, 28, 28]
print(f"标签: {first_label} (这是数字 {first_label})")

# 显示前10张图片
print("\n🖼️  显示前10张训练图片...")
fig, axes = plt.subplots(2, 5, figsize=(12, 5))
axes = axes.ravel()

for i in range(10):
    image, label = train_dataset[i]
    axes[i].imshow(image.squeeze(), cmap='gray')  # squeeze去掉通道维度
    axes[i].set_title(f'Label: {label}')
    axes[i].axis('off')

plt.tight_layout()
plt.savefig('./results/sample_images.png')  # 保存图片到results文件夹
print("✅ 前10张图片已保存到 ./results/sample_images.png")
plt.show()

print("\n🎉 数据加载成功！下一步我们可以开始构建模型了。")