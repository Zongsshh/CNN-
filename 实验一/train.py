import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from models.cnn_model import MNIST_CNN
from tqdm import tqdm
import matplotlib.pyplot as plt
import os


def train_model():
    """
    训练MNIST分类模型
    目标：准确率达到99%以上
    """
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    # 数据预处理
    transform_train = transforms.Compose([
        transforms.RandomRotation(10),  # 随机旋转
        transforms.RandomAffine(0, translate=(0.1, 0.1)),  # 随机平移
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))  # MNIST标准化参数
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    # 加载数据集
    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform_train)
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform_test)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False, num_workers=4)

    # 创建模型
    model = MNIST_CNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

    # 训练记录
    train_losses = []
    train_accuracies = []
    test_accuracies = []

    epochs = 20
    best_acc = 0.0

    print("=" * 50)
    print("开始训练...")
    print(f"模型参数量: {sum(p.numel() for p in model.parameters()):,}")
    print(f"训练轮次: {epochs}")
    print(f"批量大小: 64")
    print("=" * 50)

    for epoch in range(epochs):
        # 训练阶段
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0

        train_pbar = tqdm(train_loader, desc=f'Epoch {epoch + 1}/{epochs} [Train]')
        for batch_idx, (data, target) in enumerate(train_pbar):
            data, target = data.to(device), target.to(device)

            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            pred = output.argmax(dim=1, keepdim=True)
            correct_train += pred.eq(target.view_as(pred)).sum().item()
            total_train += target.size(0)

            if batch_idx % 100 == 0:
                train_pbar.set_postfix({
                    'Loss': f'{loss.item():.4f}',
                    'Acc': f'{100. * correct_train / total_train:.2f}%'
                })

        avg_train_loss = running_loss / len(train_loader)
        train_acc = 100. * correct_train / total_train

        # 测试阶段
        model.eval()
        correct_test = 0
        total_test = 0

        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                pred = output.argmax(dim=1, keepdim=True)
                correct_test += pred.eq(target.view_as(pred)).sum().item()
                total_test += target.size(0)

        test_acc = 100. * correct_test / total_test

        # 更新学习率
        scheduler.step(avg_train_loss)

        # 记录数据
        train_losses.append(avg_train_loss)
        train_accuracies.append(train_acc)
        test_accuracies.append(test_acc)

        # 保存最佳模型
        if test_acc > best_acc:
            best_acc = test_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_acc': best_acc,
                'train_losses': train_losses,  # 可选：保存训练历史
                'test_accuracies': test_accuracies
            }, 'best_model.pth')
            print(f"  🏆 新的最佳准确率: {test_acc:.2f}% - 模型已保存!")

        print(f"Epoch {epoch + 1}: Train Loss: {avg_train_loss:.4f}, "
              f"Train Acc: {train_acc:.2f}%, Test Acc: {test_acc:.2f}%")

    # 绘制训练曲线
    plot_training_curves(train_losses, train_accuracies, test_accuracies)

    print("=" * 50)
    print(f"训练完成! 最佳测试准确率: {best_acc:.2f}%")
    print("=" * 50)

    return model


def plot_training_curves(train_losses, train_accuracies, test_accuracies):
    """绘制训练曲线"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    # 损失曲线
    ax1.plot(train_losses, label='Training Loss', color='blue')
    ax1.set_title('Training Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.grid(True)

    # 准确率曲线
    ax2.plot(train_accuracies, label='Training Accuracy', color='green')
    ax2.plot(test_accuracies, label='Test Accuracy', color='red')
    ax2.set_title('Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig('training_curves.png', dpi=150, bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    trained_model = train_model()