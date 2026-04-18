"""
实验一最终验证脚本
功能：验证模型文件是否存在 + 生成提交清单
"""
import torch
import os
import sys


def check_experiment_results():
    """验证实验结果完整性"""

    print("🔍 正在验证实验成果...")

    # 1. 检查模型文件
    model_files = ["model.pth", "best_model.pth", "cnn_model.pth"]  # 常见命名
    model_path = None

    for filename in model_files:
        if os.path.exists(filename):
            model_path = filename
            break

    if model_path is None:
        print("❌ 未找到模型文件！请确认训练时保存了权重")
        print("💡 检查 train.py 中是否有 torch.save(...) 语句")
        return False

    print(f"✅ 找到模型文件：{model_path}")

    # 2. 尝试加载模型（仅验证格式）
    try:
        checkpoint = torch.load(model_path, map_location="cpu")
        print(f"✅ 模型文件格式正确，大小：{os.path.getsize(model_path) / 1024 / 1024:.1f} MB")
    except Exception as e:
        print(f"❌ 模型加载失败：{e}")
        return False

    # 3. 检查图像文件
    image_extensions = ['.png', '.jpg', '.jpeg', '.pdf']
    images_found = []

    for file in os.listdir('.'):
        if any(file.lower().endswith(ext) for ext in image_extensions):
            images_found.append(file)

    if images_found:
        print(f"✅ 找到图像文件：{images_found}")
    else:
        print("⚠️ 未找到训练曲线图（loss_curve.png 等）")
        print("💡 建议在 train.py 中添加 plt.savefig() 保存图像")

    # 4. 检查代码文件
    required_files = ['train.py', 'README.md', '__pycache__']
    present_files = [f for f in required_files if os.path.exists(f)]
    print(f"✅ 找到代码文件：{present_files}")

    return True


def generate_submission_list():
    """生成提交清单"""
    print("\n📋 建议提交清单：")
    print("   1. train.py                    # 主训练脚本")
    print("   2. model.pth                   # 训练好的模型权重")
    print("   3. figures/                    # 训练曲线图文件夹")
    print("      ├─ loss_curve.png")
    print("      └─ accuracy_curve.png")
    print("   4. README.md                   # 实验说明文档")
    print("   5. requirements.txt (可选)      # 依赖清单")


if __name__ == "__main__":
    success = check_experiment_results()
    if success:
        generate_submission_list()
        print("\n🎉 实验成果验证通过！可以准备提交了！")
    else:
        print("\n❌ 请先修复上述问题再提交！")
        sys.exit(1)