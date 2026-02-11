#include "tiny_infer/node.hpp"
#include <random>
#include <algorithm>
#include <immintrin.h> // SIMD
#include <fstream>

namespace tiny_infer {

// --- Linear 实现 ---

Linear::Linear(size_t in_features, size_t out_features) 
    : m_weights({out_features, in_features}), 
      m_bias({out_features})
{
    // 初始化权重 (简单随机初始化，模拟 PyTorch xavier_uniform)
    // 实际项目中应该从文件加载
    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dist(-0.1f, 0.1f);

    size_t weight_size = m_weights.shape()[0] * m_weights.shape()[1];
    float* weight_ptr = m_weights.data();
    for (size_t i = 0; i < weight_size; ++i) weight_ptr[i] = dist(gen);
    
    size_t bias_size = m_bias.shape()[0];
    float* bias_ptr = m_bias.data();
    for (size_t i = 0; i < bias_size; ++i) bias_ptr[i] = 0.01f;
}

Tensor Linear::forward(const Tensor& input) {
    // Input: [Batch, In], Weights: [Out, In]
    // Output = Input * Weights^T (因为我们的 matmul 内部做了 transpose，
    // 但通常全连接层的权重存储方式就是 [Out, In]，直接把 Weights 当做 B 传进去即可)
    
    // 注意：这里为了利用我们写好的 C = A * B，我们需要处理一下逻辑。
    // 我们的 matmul 实现是 C = A * B (且内部转置了 B)。
    // 全连接层公式通常是 Y = X * W^T + b。
    // 如果 m_weights 是 [Out, In]，直接传给我们的 matmul (它会转置 B -> [In, Out])
    // 结果就是 [Batch, In] * [In, Out] = [Batch, Out]。完美！
    
    Tensor output = Tensor::matmul(input, m_weights);

    // 加上 Bias (广播机制 Broadcast)
    // Bias 是 [Out]，需要加到 output 的每一行上
    size_t batch_size = output.shape()[0];
    size_t out_features = output.shape()[1];
    float* out_ptr = output.data();
    const float* bias_ptr = m_bias.data();

    for (size_t i = 0; i < batch_size; ++i) {
        for (size_t j = 0; j < out_features; ++j) {
            out_ptr[i * out_features + j] += bias_ptr[j];
        }
    }

    return output;
}

void Linear::load_params(const std::string& w_path, const std::string& b_path) {
    auto load_bin = [](const std::string& path, Tensor& t) {
        std::ifstream file(path, std::ios::binary | std::ios::ate);
        if (!file) throw std::runtime_error("Cannot open file: " + path);
        
        std::streamsize size = file.tellg();
        file.seekg(0, std::ios::beg);

        // 安全计算总元素数量
        size_t total_elements = 1;
        for (auto s : t.shape()) total_elements *= s;
        size_t expected_bytes = total_elements * sizeof(float);

        if (size != (std::streamsize)expected_bytes) {
            throw std::runtime_error("File size mismatch for " + path);
        }

        file.read(reinterpret_cast<char*>(t.data()), size);
    };

    load_bin(w_path, m_weights);
    load_bin(b_path, m_bias);
}
// --- ReLU 实现 (SIMD 优化版) ---

Tensor ReLU::forward(const Tensor& input) {
    Tensor output = input; // 拷贝一份 (实际框架会支持 in-place)
    float* ptr = output.data();
    size_t size = output.shape()[0] * output.shape()[1];

    size_t i = 0;
    __m256 zeros = _mm256_setzero_ps(); // 创建全0向量

    // AVX2 处理
    for (; i + 8 <= size; i += 8) {
        __m256 v = _mm256_loadu_ps(ptr + i);
        // max(0, v)
        v = _mm256_max_ps(zeros, v);
        _mm256_storeu_ps(ptr + i, v);
    }

    // 处理剩余部分
    for (; i < size; ++i) {
        if (ptr[i] < 0) ptr[i] = 0;
    }

    return output;
}

} // namespace tiny_infer