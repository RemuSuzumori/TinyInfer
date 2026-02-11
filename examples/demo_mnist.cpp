#include <iostream>
#include <vector>
#include <fstream>
#include "tiny_infer/node.hpp" // 使用你的 node.hpp

using namespace tiny_infer;

int main() {
    std::cout << "=== TinyInfer vs PyTorch Consistency Check ===" << std::endl;

    // 1. 定义与 Python 完全一致的架构 (784 -> 128 -> 10)
    Linear fc1(784, 128);
    ReLU relu;
    Linear fc2(128, 10);

    // 2. 加载导出的权重文件 (路径需根据你的 exe 位置调整，这里假设在项目根目录运行)
    try {
        fc1.load_params("models/fc1_weights.bin", "models/fc1_bias.bin");
        fc2.load_params("models/fc2_weights.bin", "models/fc2_bias.bin");
        std::cout << "[Step 1] Weights loaded successfully." << std::endl;
    } catch (...) {
        std::cout << "❌ Error: Cannot find .bin files in 'models/' folder!" << std::endl;
        return -1;
    }

    // 3. 加载 PyTorch 生成的测试输入 (input.bin)
    Tensor input({1, 784});
    std::ifstream ifs("models/input.bin", std::ios::binary);
    ifs.read(reinterpret_cast<char*>(input.data()), 784 * sizeof(float));
    std::cout << "[Step 2] Input data loaded." << std::endl;

    // 4. 执行推理
    auto h1 = fc1.forward(input);
    auto h2 = relu.forward(h1);
    auto output = fc2.forward(h2);

    // 5. 寻找最大概率的索引 (Argmax)
    float* results = output.data();
    int predicted_digit = 0;
    float max_score = results[0];

    for (int i = 1; i < 10; ++i) {
        if (results[i] > max_score) {
            max_score = results[i];
            predicted_digit = i;
        }
    }

    std::cout << "\n" << "===============================" << std::endl;
    std::cout << "Final Prediction: " << predicted_digit << std::endl;
    std::cout << "Confidence Score: " << max_score << std::endl;
    std::cout << "===============================" << std::endl;

    // 打印一下分布，看看模型有多确信
    std::cout << "Probability Distribution:" << std::endl;
    for (int i = 0; i < 10; ++i) {
        printf("[%d]: %.4f  ", i, results[i]);
    }
    std::cout << std::endl;
}