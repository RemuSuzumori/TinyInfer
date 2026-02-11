#include <iostream>
#include <vector>
#include <fstream>
#include <iomanip>
#include "tiny_infer/node.hpp" 

using namespace tiny_infer;

int main() {
    std::cout << "=== TinyInfer Consistency Verification ===" << std::endl;

    // 1. 架构对齐: 784 -> 128 -> 10 (必须与 Python 一致)
    Linear fc1(784, 128);
    ReLU relu;
    Linear fc2(128, 10);

    // 2. 加载 Python 导出的权重
    try {
        // 假设你在项目根目录运行，models 文件夹就在旁边
        fc1.load_params("models/fc1_weights.bin", "models/fc1_bias.bin");
        fc2.load_params("models/fc2_weights.bin", "models/fc2_bias.bin");
        std::cout << "[OK] Parameters loaded from models/." << std::endl;
    } catch (...) {
        std::cout << "[ERROR] Cannot find .bin files! Check 'models' folder." << std::endl;
        return -1;
    }

    // 3. 加载 Python 导出的相同输入 (input.bin)
    Tensor input({1, 784});
    std::ifstream ifs("models/input.bin", std::ios::binary);
    if (!ifs) return -1;
    ifs.read(reinterpret_cast<char*>(input.data()), 784 * sizeof(float));

    // 4. 执行推理
    auto h1 = fc1.forward(input);
    auto h2 = relu.forward(h1);
    auto output = fc2.forward(h2);

    // 5. 打印结果对比
    std::cout << "\nC++ Inference Result (First 5):" << std::endl;
    for (int i = 0; i < 5; ++i) {
        std::cout << std::fixed << std::setprecision(8) << output.data()[i] << " ";
    }
    std::cout << "\n\nCompare this with your PyTorch Reference Output!" << std::endl;

    return 0;
}