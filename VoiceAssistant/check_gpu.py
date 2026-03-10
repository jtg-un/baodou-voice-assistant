import torch

print("-" * 30)
if torch.cuda.is_available():
    print("✅ 恭喜！GPU 加速环境已就绪")
    print(f"🎮 显卡型号: {torch.cuda.get_device_name(0)}")
    print(f"🚀 CUDA 版本: {torch.version.cuda}")
    print(f"📦 Torch 版本: {torch.__version__}")
else:
    print("❌ 失败：当前运行的仍是 CPU 版本")
    print("💡 建议：重新执行阿里云的安装命令，并检查是否有报错")
print("-" * 30)