import numpy as np
from PIL import Image
import matplotlib.pyplot as plt # 新增
import os

def prepare_image(image_path):
    # 1. 打开并转为灰度 ('L')
    img = Image.open(image_path).convert('L')
    
    # 2. 缩放到 28x28
    img = img.resize((28, 28))
    
    # 3. 转换为数组供查看和保存
    data = np.array(img).astype(np.float32) / 255.0
    data = 1.0 - data
    data = (data - 0.1307) / 0.3081  # 增加这一行，对齐 MNIST 训练分布

    
    # --- 查看部分开始 ---
    print(f">>> 正在预览 28x28 灰度图...")
    plt.imshow(data, cmap='gray') # 使用灰色映射
    plt.title(f"Processed: {image_path} (28x28)")
    plt.show() # 这会弹出一个窗口显示图片
    # --- 查看部分结束 ---

    # 4. 展平并保存二进制
    data.reshape(1, 784).tofile("models/input.bin")
    print(f"✅ 数据已保存。")

if __name__ == "__main__":
    prepare_image("test.png")