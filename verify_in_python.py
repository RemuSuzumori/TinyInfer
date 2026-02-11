import torch
import torch.nn as nn
import numpy as np
import os

# 1. 定义完全相同的网络结构
class TinyNet(nn.Module):
    def __init__(self):
        super(TinyNet, self).__init__()
        self.fc1 = nn.Linear(784, 128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

def verify():
    print(">>> [Python 验证模式] 启动...")
    model = TinyNet()
    
    # 2. 手动加载和你 C++ 一模一样的 .bin 权重
    # 【核心修改区】
    # 因为磁盘上的 .bin 文件已经被我们统一改成了 [In, Out] 格式
    # 所以我们先按 [In, Out] 读取，然后用 .T 翻转回 PyTorch 想要的 [Out, In] 格式
    try:
        # FC1 磁盘是 [784, 128]，读取后转置给 PyTorch [128, 784]
        fc1_w = np.fromfile("models/fc1_weights.bin", dtype=np.float32).reshape(784, 128).T
        fc1_b = np.fromfile("models/fc1_bias.bin", dtype=np.float32)
        
        # FC2 磁盘是 [128, 10]，读取后转置给 PyTorch [10, 128]
        fc2_w = np.fromfile("models/fc2_weights.bin", dtype=np.float32).reshape(128, 10).T
        fc2_b = np.fromfile("models/fc2_bias.bin", dtype=np.float32)
    except FileNotFoundError:
        print("❌ 找不到权重文件，请先运行 train_and_export.py")
        return

    # 暴力赋值给 PyTorch 模型 (使用 .copy() 保证内存连续性)
    with torch.no_grad():
        model.fc1.weight.data = torch.from_numpy(fc1_w.copy())
        model.fc1.bias.data = torch.from_numpy(fc1_b)
        model.fc2.weight.data = torch.from_numpy(fc2_w.copy())
        model.fc2.bias.data = torch.from_numpy(fc2_b)
    
    model.eval()

    # 3. 加载 input.bin (C++ 吃的那个文件)
    try:
        input_data = np.fromfile("models/input.bin", dtype=np.float32).reshape(1, 784)
    except FileNotFoundError:
        print("❌ 找不到 models/input.bin")
        return

    # 4. 推理
    output = model(torch.from_numpy(input_data))
    
    # 5. 打印结果
    probs = output.detach().numpy()[0]
    pred = np.argmax(probs)
    
    print("\n" + "="*30)
    print(f"PyTorch 认为这张图是: 【 {pred} 】")
    print("="*30)
    print("分数分布:")
    for i, p in enumerate(probs):
        print(f"[{i}]: {p:.4f}", end="  ")
        if i == 4: print() # 换行
    print("\n")

if __name__ == "__main__":
    verify()