import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os

# 1. 定义和 C++ 一模一样的网络结构
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # Input: 28x28=784, Hidden: 128, Output: 10
        self.fc1 = nn.Linear(784, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = torch.flatten(x, 1)
        
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def export_weights():
    # 模拟训练好的模型 (这里我们下载预训练权重，或者快速训练一下)
    # 为了演示简单，我们先用随机初始化的权重，但保存为文件
    # 如果你想用真模型，可以去网上找个 MNIST pth 加载
    model = Net()
    model.eval()

    print("Exporting weights to 'models/' directory...")
    os.makedirs("models", exist_ok=True)

    # --- 导出第一层 (fc1) ---
    # PyTorch: [128, 784] -> C++ Expects: [784, 128]
    fc1_w = model.fc1.weight.detach().numpy().T 
    fc1_b = model.fc1.bias.detach().numpy()
    
    fc1_w.astype(np.float32).tofile("models/fc1_weights.bin")
    fc1_b.astype(np.float32).tofile("models/fc1_bias.bin")

    # --- 导出第二层 (fc2) ---
    # PyTorch: [10, 128] -> C++ Expects: [128, 10]
    fc2_w = model.fc2.weight.detach().numpy().T
    fc2_b = model.fc2.bias.detach().numpy()

    fc2_w.astype(np.float32).tofile("models/fc2_weights.bin")
    fc2_b.astype(np.float32).tofile("models/fc2_bias.bin")

    print(f"FC1 shape: {fc1_w.shape}, saved.")
    print(f"FC2 shape: {fc2_w.shape}, saved.")
    print("Done!")

    # 顺便导出一个简单的输入数据用于测试
    dummy_input = torch.randn(1, 784).numpy().astype(np.float32)
    dummy_input.tofile("models/input.bin")
    
    # 计算 PyTorch 的标准输出，用于去 C++ 对答案
    with torch.no_grad():
        out = model(torch.tensor(dummy_input))
        print("\nPyTorch Reference Output:")
        print(out.numpy()[0])

if __name__ == "__main__":
    export_weights()