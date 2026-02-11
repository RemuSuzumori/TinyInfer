import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import os

# 1. 定义与你 C++ 结构完全一致的网络 (784 -> 128 -> 10)
class TinyNet(nn.Module):
    def __init__(self):
        super(TinyNet, self).__init__()
        self.fc1 = nn.Linear(784, 128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = x.view(-1, 784)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

def train():
    print(">>> 正在准备数据并开始 1 分钟快速训练...")
    # MNIST 标准预处理：转换为 Tensor 并标准化
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('./data', train=True, download=True, transform=transform),
        batch_size=64, shuffle=True)

    model = TinyNet()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()

    # 只训练 1 个 Epoch，足够在简单的 MLP 上达到 90%+ 准确率
    model.train()
    for epoch in range(1, 6): # 训练 5 个 Epoch
        print(f"\n--- 开始第 {epoch} 轮训练 ---")
        for batch_idx, (data, target) in enumerate(train_loader):
            optimizer.zero_grad()
            loss = criterion(model(data), target)
            loss.backward()
            optimizer.step()
            if batch_idx % 400 == 0:
                print(f"进度: [{batch_idx * 64}/60000] Loss: {loss.item():.4f}")

    print(">>> 训练完成！开始导出权重...")
    
    # 2. 导出逻辑 (对应你 C++ 的 load_params)
    # ----- 请将原来导出的部分替换为以下代码 -----
    print(">>> 训练完成！开始导出权重 (已执行转置适配 C++ 引擎)...")
    os.makedirs("models", exist_ok=True)
    
# 位于 train_and_export.py 结尾
    # 导出时务必转置，以匹配 C++ [In, Out] 的预期形状
    model.fc1.weight.detach().numpy().T.copy().tofile("models/fc1_weights.bin")
    model.fc1.bias.detach().numpy().tofile("models/fc1_bias.bin")
    
    model.fc2.weight.detach().numpy().T.copy().tofile("models/fc2_weights.bin")
    model.fc2.bias.detach().numpy().tofile("models/fc2_bias.bin")
    
    print("✅ 真实权重已保存至 models/ 目录。")

if __name__ == "__main__":
    train()