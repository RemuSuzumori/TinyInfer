import torch
import torch.nn as nn
import numpy as np
import os

# -----------------------------
# 1) 定义网络结构
# -----------------------------
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


# -----------------------------
# 2) 权重加载（可被 UI/脚本复用）
# -----------------------------
def load_model(
    model_dir="models",
    device="cpu"
):
    """
    加载 TinyNet + 从 .bin 加载权重，并返回 eval 模式的 model。
    如果文件不存在会抛出 FileNotFoundError。
    """
    model = TinyNet().to(device)

    fc1_w_path = os.path.join(model_dir, "fc1_weights.bin")
    fc1_b_path = os.path.join(model_dir, "fc1_bias.bin")
    fc2_w_path = os.path.join(model_dir, "fc2_weights.bin")
    fc2_b_path = os.path.join(model_dir, "fc2_bias.bin")

    # 你的磁盘格式：fc1_w [784,128]、fc2_w [128,10]，PyTorch 需要转置成 [out,in]
    fc1_w = np.fromfile(fc1_w_path, dtype=np.float32).reshape(784, 128).T
    fc1_b = np.fromfile(fc1_b_path, dtype=np.float32)

    fc2_w = np.fromfile(fc2_w_path, dtype=np.float32).reshape(128, 10).T
    fc2_b = np.fromfile(fc2_b_path, dtype=np.float32)

    with torch.no_grad():
        model.fc1.weight.data = torch.from_numpy(fc1_w.copy()).to(device)
        model.fc1.bias.data = torch.from_numpy(fc1_b).to(device)
        model.fc2.weight.data = torch.from_numpy(fc2_w.copy()).to(device)
        model.fc2.bias.data = torch.from_numpy(fc2_b).to(device)

    model.eval()
    return model


# -----------------------------
# 3) 推理接口（UI 最常用）
# -----------------------------
@torch.no_grad()
def infer_from_array(model, x_784):
    """
    x_784: np.ndarray，形状可为 (784,) 或 (1,784)
    返回: (pred:int, logits:np.ndarray shape(10,))
    """
    x = np.asarray(x_784, dtype=np.float32)
    if x.ndim == 1:
        if x.shape[0] != 784:
            raise ValueError(f"x_784 length must be 784, got {x.shape[0]}")
        x = x.reshape(1, 784)
    elif x.ndim == 2:
        if x.shape != (1, 784):
            raise ValueError(f"x_784 shape must be (1,784), got {x.shape}")
    else:
        raise ValueError(f"x_784 must be 1D or 2D, got ndim={x.ndim}")

    out = model(torch.from_numpy(x))
    logits = out.detach().cpu().numpy()[0]
    pred = int(np.argmax(logits))
    return pred, logits


@torch.no_grad()
def infer_from_input_bin(model, input_bin_path="models/input.bin"):
    """
    读取 C++ 用的 models/input.bin -> 推理
    返回: (pred:int, logits:np.ndarray shape(10,))
    """
    x = np.fromfile(input_bin_path, dtype=np.float32).reshape(1, 784)
    return infer_from_array(model, x)


# -----------------------------
# 4) 命令行验证（保持你原来的输出风格）
# -----------------------------
def verify():
    print(">>> [Python 验证模式] 启动...")

    try:
        model = load_model(model_dir="models", device="cpu")
    except FileNotFoundError as e:
        print("❌ 找不到权重文件，请先运行 train_and_export.py 或确认 models/ 下文件齐全")
        print(f"   缺失: {e}")
        return

    try:
        pred, probs = infer_from_input_bin(model, "models/input.bin")
    except FileNotFoundError:
        print("❌ 找不到 models/input.bin")
        return

    print("\n" + "=" * 30)
    print(f"PyTorch 认为这张图是: 【 {pred} 】")
    print("=" * 30)
    print("分数分布:")
    for i, p in enumerate(probs):
        print(f"[{i}]: {p:.4f}", end="  ")
        if i == 4:
            print()
    print("\n")


if __name__ == "__main__":
    verify()
