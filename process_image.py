import numpy as np
from PIL import Image, ImageOps
import os

def mnistify(image_path, out_bin="models/input.bin", out_png="models/input_preview.png",
             invert_auto=True, binarize=True):
    os.makedirs(os.path.dirname(out_bin), exist_ok=True)

    # 1) 读图 -> 灰度
    img = Image.open(image_path).convert("L")

    # 2) 自动判断是否需要反色：目标是“黑底白字”（笔画亮）
    # 判断依据：平均灰度高 -> 可能是白底黑字，需要反色
    arr0 = np.array(img, dtype=np.uint8)
    if invert_auto:
        if arr0.mean() > 127:
            img = ImageOps.invert(img)

    # 3) 去噪 + 二值化（可选，但对拍照/扫描很有用）
    # 简单阈值：先用中值滤波再阈值（不依赖opencv）
    # 你也可以把阈值调成 80~180 试试
    if binarize:
        # PIL 自带的点运算阈值
        img = img.point(lambda p: 255 if p > 80 else 0)

    # 4) 裁剪到数字本体（去掉大片空白）
    arr = np.array(img, dtype=np.uint8)
    ys, xs = np.where(arr > 0)  # 白色笔画
    if len(xs) == 0 or len(ys) == 0:
        raise RuntimeError("没检测到笔画（图像可能太暗/阈值不合适/反色不对）")

    x0, x1 = xs.min(), xs.max()
    y0, y1 = ys.min(), ys.max()
    img = img.crop((x0, y0, x1 + 1, y1 + 1))

    # 5) 等比缩放：最长边 -> 20 像素（MNIST 经典做法）
    w, h = img.size
    scale = 20.0 / max(w, h)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    img = img.resize((new_w, new_h), resample=Image.BILINEAR)

    # 6) padding 到 28×28，并“居中”
    canvas = Image.new("L", (28, 28), 0)
    left = (28 - new_w) // 2
    top  = (28 - new_h) // 2
    canvas.paste(img, (left, top))

    # 7) 生成网络输入：ToTensor -> [0,1] -> 标准化
    data = np.array(canvas, dtype=np.float32) / 255.0
    data = (data - 0.1307) / 0.3081

    # 8) 保存预览图（保存“未标准化”的更直观）
    preview = (np.array(canvas, dtype=np.uint8))
    Image.fromarray(preview).save(out_png)

    # 9) 保存 bin（float32, 784）
    data.reshape(1, 784).astype(np.float32).tofile(out_bin)

    print("✅ MNIST化完成")
    print(f"  - preview: {out_png}")
    print(f"  - input.bin: {out_bin}")
    print(f"  - invert_auto={invert_auto}, binarize={binarize}, crop=({x0},{y0})-({x1},{y1})")

if __name__ == "__main__":
    mnistify("test.png")
