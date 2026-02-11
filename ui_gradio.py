import gradio as gr
from PIL import Image
import numpy as np

from process_image import mnistify
from verify_in_python import load_model, infer_from_input_bin

MODEL = load_model("models")


def _run_pipeline_from_pil(pil_img: Image.Image):
    """
    输入：PIL Image
    输出：pred(str), scores(dict), preview_path(str)
    """
    # 1) 保存临时图
    tmp = "._ui_tmp.png"
    pil_img.convert("L").save(tmp)

    # 2) MNIST 化 -> 写 input.bin + preview
    mnistify(
        tmp,
        out_bin="models/input.bin",
        out_png="models/input_preview.png",
        invert_auto=True,
        binarize=True
    )

    # 3) 推理
    pred, logits = infer_from_input_bin(MODEL, "models/input.bin")

    scores = {str(i): float(logits[i]) for i in range(10)}
    return str(pred), scores, "models/input_preview.png"


def predict_upload(img: Image.Image):
    if img is None:
        return "", {}, None
    return _run_pipeline_from_pil(img)


def _to_pil_image(obj):
    """
    把 Gradio Sketchpad 可能返回的类型统一转成 PIL.Image：
    - PIL.Image
    - numpy array
    - dict（常见：{"image":..., "mask":...} / {"composite":...} / {"background":...}）
    """
    if obj is None:
        return None

    # 1) dict：取出真正的图像字段
    if isinstance(obj, dict):
        # 常见 key 优先级：image > composite > background > layers[0]
        for key in ["image", "composite", "background"]:
            if key in obj and obj[key] is not None:
                obj = obj[key]
                break
        else:
            # 有些版本是 layers 列表
            if "layers" in obj and obj["layers"]:
                obj = obj["layers"][0]
            else:
                raise ValueError(f"Sketchpad returned dict but no known image field: keys={list(obj.keys())}")

    # 2) 已经是 PIL
    if isinstance(obj, Image.Image):
        return obj

    # 3) numpy array
    if isinstance(obj, np.ndarray):
        arr = obj
    else:
        # 兜底：尝试转成 np.array
        arr = np.array(obj)

    # 4) 处理维度：H×W、H×W×3、H×W×4
    if arr.ndim == 2:
        return Image.fromarray(arr.astype(np.uint8), mode="L")
    if arr.ndim == 3 and arr.shape[2] in (3, 4):
        # RGBA -> RGB（画板有时带 alpha）
        arr = arr[:, :, :3]
        return Image.fromarray(arr.astype(np.uint8), mode="RGB")

    raise ValueError(f"Unsupported sketch image array shape: {arr.shape}")


def predict_sketch(sketch_obj):
    if sketch_obj is None:
        return "", {}, None

    pil_img = _to_pil_image(sketch_obj)
    if pil_img is None:
        return "", {}, None

    return _run_pipeline_from_pil(pil_img)

    """
    Gradio Sketchpad 可能给你：
      - PIL.Image
      - numpy array
    我们都统一转 PIL 再跑。
    """
    if sketch_img is None:
        return "", {}, None

    # Sketchpad 常见输出是 numpy (H,W,3) 或 (H,W,4)
    if isinstance(sketch_img, np.ndarray):
        if sketch_img.ndim == 3 and sketch_img.shape[2] in (3, 4):
            sketch_img = Image.fromarray(sketch_img.astype(np.uint8))
        elif sketch_img.ndim == 2:
            sketch_img = Image.fromarray(sketch_img.astype(np.uint8))
        else:
            raise ValueError(f"Unexpected sketch ndarray shape: {sketch_img.shape}")

    if not isinstance(sketch_img, Image.Image):
        # 兜底
        sketch_img = Image.fromarray(np.array(sketch_img).astype(np.uint8))

    return _run_pipeline_from_pil(sketch_img)


with gr.Blocks(title="TinyInfer MNIST UI") as demo:
    gr.Markdown("# TinyInfer MNIST Demo")
    gr.Markdown("支持两种输入：**上传图片** 或 **在画板手写**。两者都走同一条 mnistify → 推理链路。")

    with gr.Tabs():
        with gr.Tab("📁 上传图片识别"):
            up_in = gr.Image(type="pil", label="上传图片")
            up_btn = gr.Button("识别")
            up_pred = gr.Textbox(label="预测数字")
            up_scores = gr.Label(label="10类分数(logits)")
            up_prev = gr.Image(type="filepath", label="MNIST化预览(28x28)")

            up_btn.click(fn=predict_upload, inputs=up_in, outputs=[up_pred, up_scores, up_prev])

        with gr.Tab("✍️ 手写识别"):
            gr.Markdown("在下面画板写数字（建议写大一点、居中），然后点识别。")

            # ✅ 画板：用户手写
            sketch = gr.Sketchpad(label="手写区域", height=280, width=280)

            with gr.Row():
                sk_btn = gr.Button("识别")
                sk_clear = gr.Button("清空")

            sk_pred = gr.Textbox(label="预测数字")
            sk_scores = gr.Label(label="10类分数(logits)")
            sk_prev = gr.Image(type="filepath", label="MNIST化预览(28x28)")

            sk_btn.click(fn=predict_sketch, inputs=sketch, outputs=[sk_pred, sk_scores, sk_prev])
            sk_clear.click(fn=lambda: None, inputs=None, outputs=sketch)

demo.launch()
