import os
import sys
import glob

# 1. 暴力自动查找 .pyd 文件
# 我们直接在 build 文件夹里搜寻那个生成的二进制模块
search_path = os.path.join(os.getcwd(), "build", "**", "tiny_infer_py*.pyd")
found_files = glob.glob(search_path, recursive=True)

if not found_files:
    print("❌ 找不到生成的 .pyd 模块！请先确保执行了 cmake --build build --config Release")
    sys.exit(1)

# 把找到模块所在的目录加入 Python 搜索路径
module_dir = os.path.dirname(found_files[0])
sys.path.append(module_dir)
print(f"✅ 找到模块路径: {module_dir}")

try:
    import tiny_infer_py as ti  # type: ignore
    print("✅ 模块导入成功！")
    
    # 执行一个简单的测试
    fc1 = ti.Linear(128, 256)
    input_data = ti.Tensor([1, 128])
    input_data.fill(0.5)
    
    print("--- 正在执行推理 ---")
    output = fc1.forward(input_data)
    print(f"推理成功，输出维度: {output.shape()}")
    print(f"前 3 个数值: {[output.at(0, i) for i in range(3)]}")
    
except Exception as e:
    print(f"❌ 运行失败: {e}")
    print(f"提示：请检查你的 Python 版本是否为 3.12 (你的模块是为 3.12 编译的)")