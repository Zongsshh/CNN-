import torch
print("🔍 PyTorch 版本:", torch.__version__)
print("🔍 CUDA 可用:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("✅ GPU 设备名:", torch.cuda.get_device_name(0))
    print("✅ 显存总量:", torch.cuda.get_device_properties(0).total_memory / 1024**3, "GB")
else:
    print("❌ GPU 不可用！")
    # 尝试诊断
    try:
        import ctypes
        ctypes.CDLL("cudart64_110.dll")  # 尝试加载 CUDA 运行时
        print("⚠️ cudart64_110.dll 可加载，但 PyTorch 仍报错？可能是 PATH 问题")
    except Exception as e:
        print("❌ 缺少 CUDA 运行时 DLL:", e)