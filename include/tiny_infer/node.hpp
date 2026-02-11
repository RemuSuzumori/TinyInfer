#pragma once
#include "tiny_infer/tensor.hpp"
#include <vector>

#include <memory>
#include <string>

namespace tiny_infer {

// 所有层的基类
class Layer {
public:
    virtual ~Layer() = default;

    // 核心接口：前向传播
    // 输入一个 Tensor，输出一个 Tensor
    virtual Tensor forward(const Tensor& input) = 0;

    // 获取层的名字（调试用）
    virtual std::string type() const = 0;
};

// 1. 全连接层 (Linear / Dense)
// 公式：Output = Input * Weights^T + Bias
class Linear : public Layer {
public:
    // in_features: 输入维度, out_features: 输出维度
    Linear(size_t in_features, size_t out_features);

    Tensor forward(const Tensor& input) override;
    std::string type() const override { return "Linear"; }

    // 获取权重（用于加载模型参数）
    Tensor& weights() { return m_weights; }
    Tensor& bias() { return m_bias; }
    // 加载二进制权重文件
    void load_params(const std::string& w_path, const std::string& b_path);

private:
    Tensor m_weights; // 形状 [out_features, in_features]
    Tensor m_bias;    // 形状 [out_features]
};

// 2. 激活函数层 (ReLU)
// 公式：Output = max(0, Input)
class ReLU : public Layer {
public:
    Tensor forward(const Tensor& input) override;
    std::string type() const override { return "ReLU"; }
};

} // namespace tiny_infer