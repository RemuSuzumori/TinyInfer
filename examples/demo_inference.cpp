#include <iostream>
#include <vector>
#include <chrono>
#include <iomanip>
#include "../include/tiny_infer/tensor.hpp"
#include "../include/tiny_infer/node.hpp"

using namespace tiny_infer;

int main() {
   std::cout << "--- TinyInfer Demo Start ---" << std::endl;

    // 1. 定义网络结构
    // 假设输入是 128维，隐藏层 256维，输出 10类
    size_t batch_size = 4;
    size_t in_dim = 128;
    size_t hidden_dim = 256;
    size_t out_dim = 10;

    std::cout << "Model Architecture:" << std::endl;
    std::cout << "  Linear (" << in_dim << " -> " << hidden_dim << ")" << std::endl;
    std::cout << "  ReLU" << std::endl;
    std::cout << "  Linear (" << hidden_dim << " -> " << out_dim << ")" << std::endl;

    Linear fc1(in_dim, hidden_dim);
    ReLU relu;
    Linear fc2(hidden_dim, out_dim);

    // 2. 准备输入数据 (Batch Size = 4)
    Tensor input({batch_size, in_dim});
    input.fill(0.5f); // 填充一些假数据

    std::cout << "\nRunning Inference..." << std::endl;

    // 3. 执行推理并计时
    auto start = std::chrono::high_resolution_clock::now();

    // Layer 1
    Tensor h1 = fc1.forward(input);
    
    // Activation
    Tensor h2 = relu.forward(h1);
    
    // Layer 2
    Tensor output = fc2.forward(h2);

    auto end = std::chrono::high_resolution_clock::now();
    double elapsed_us = std::chrono::duration<double, std::micro>(end - start).count();

    std::cout << "Inference finished in " << elapsed_us << " us (" << elapsed_us/1000.0 << " ms)" << std::endl;

    // 4. 打印输出结果 (检查是否计算出东西了)
    std::cout << "\nOutput Tensor (First sample):" << std::endl;
    for (size_t j = 0; j < out_dim; ++j) {
        std::cout << std::fixed << std::setprecision(4) << output.at(0, j) << " ";
    }
    std::cout << std::endl;

    return 0;
}