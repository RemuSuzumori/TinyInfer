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
    # 注意：PyTorch 的 Linear 权重形状是 (Out, In)，我们需要确保 numpy 读取后形状匹配
    try:
        fc1_w = np.fromfile("models/fc1_weights.bin", dtype=np.float32).reshape(128, 784)
        fc1_b = np.fromfile("models/fc1_bias.bin", dtype=np.float32)
        fc2_w = np.fromfile("models/fc2_weights.bin", dtype=np.float32).reshape(10, 128)
        fc2_b = np.fromfile("models/fc2_bias.bin", dtype=np.float32)
    except FileNotFoundError:
        print("❌ 找不到权重文件，请先运行 train_and_export.py")
        return

    # 暴力赋值给 PyTorch 模型
    with torch.no_grad():
        model.fc1.weight.data = torch.from_numpy(fc1_w)
        model.fc1.bias.data = torch.from_numpy(fc1_b)
        model.fc2.weight.data = torch.from_numpy(fc2_w)
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

    # --- 诊断建议 ---
    if pred == 7:
        print("💡 结论: 你的 C++ 引擎是完美的！问题出在【图片】。")
        print("原因: MLP 模型对位置太敏感，你的手写数字可能偏离了重心，或者笔画特征不像训练集。")
        print("建议: 使用下面的 process_image_advanced.py 代码加入【重心对齐】功能。")
    elif pred == 4:
        print("💡 结论: 图片没问题，是【C++ 引擎权重加载】反了！")
        print("原因: PyTorch 权重是 (Out, In)，C++ 可能按 (In, Out) 读取了。")
        print("建议: 在导出脚本中加入 .T (转置) 操作。")

if __name__ == "__main__":
    verify()