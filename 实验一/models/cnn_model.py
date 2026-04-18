import torch
import torch.nn as nn
import torch.nn.functional as F


class MNIST_CNN(nn.Module):
    """
    超强CNN：目标准确率 99.0%+
    4个卷积块 = 8个卷积层 + BatchNorm + Dropout
    """

    def __init__(self):
        super(MNIST_CNN, self).__init__()

        # 卷积块1: 1→32通道
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.dropout1 = nn.Dropout2d(0.25)

        # 卷积块2: 32→64通道
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.dropout2 = nn.Dropout2d(0.25)

        # 卷积块3: 64→128通道
        self.conv5 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn5 = nn.BatchNorm2d(128)
        self.conv6 = nn.Conv2d(128, 128, kernel_size=3, padding=1)  # 新增！
        self.bn6 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(2, 2)
        self.dropout3 = nn.Dropout2d(0.25)

        # 卷积块4: 128→256通道（新增更深层）
        self.conv7 = nn.Conv2d(128, 256, kernel_size=3, padding=1)  # 新增！
        self.bn7 = nn.BatchNorm2d(256)
        self.conv8 = nn.Conv2d(256, 256, kernel_size=3, padding=1)  # 新增！
        self.bn8 = nn.BatchNorm2d(256)
        self.pool4 = nn.MaxPool2d(2, 2)  # 3×3 → 1×1 (因为3//2=1)
        self.dropout4 = nn.Dropout2d(0.25)

        # 全连接层
        self.fc1 = nn.Linear(256 * 1 * 1, 512)  # 256×1×1=256 → 512
        self.bn_fc1 = nn.BatchNorm1d(512)
        self.dropout_fc1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, 128)  # 512 → 128
        self.bn_fc2 = nn.BatchNorm1d(128)
        self.dropout_fc2 = nn.Dropout(0.5)
        self.fc3 = nn.Linear(128, 10)  # 128 → 10（10个类别）

    def forward(self, x):  # 修复：添加了x参数
        # 卷积块1
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool1(x)
        x = self.dropout1(x)

        # 卷积块2
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = self.pool2(x)
        x = self.dropout2(x)

        # 卷积块3
        x = F.relu(self.bn5(self.conv5(x)))
        x = F.relu(self.bn6(self.conv6(x)))
        x = self.pool3(x)
        x = self.dropout3(x)

        # 卷积块4（新增）
        x = F.relu(self.bn7(self.conv7(x)))
        x = F.relu(self.bn8(self.conv8(x)))
        x = self.pool4(x)  # 3×3经过maxpool变成1×1
        x = self.dropout4(x)

        # 展平 + 全连接
        x = x.view(x.size(0), -1)  # [batch, 256]
        x = F.relu(self.bn_fc1(self.fc1(x)))
        x = self.dropout_fc1(x)
        x = F.relu(self.bn_fc2(self.fc2(x)))
        x = self.dropout_fc2(x)
        x = self.fc3(x)

        return x


# 快速测试模型
if __name__ == "__main__":
    model = MNIST_CNN()
    dummy_input = torch.randn(4, 1, 28, 28)
    output = model(dummy_input)
    print("✅ 超强CNN模型测试通过！")
    print(f"输入形状: {dummy_input.shape}")
    print(f"输出形状: {output.shape}")
    print(f"模型参数量: {sum(p.numel() for p in model.parameters()):,}")