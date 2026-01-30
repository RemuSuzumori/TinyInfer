#include <gtest/gtest.h>
#include "tiny_infer/node.hpp"

TEST(ModelTest, SimpleInference) {
    // 模拟一个简单的神经网络： Input -> Linear -> ReLU -> Output
    // Batch Size = 4, Input Features = 128
    tiny_infer::Tensor input({4, 128});
    input.fill(1.0f);

    // 定义层
    // Layer 1: 128 -> 64
    tiny_infer::Linear fc1(128, 64);
    // Layer 2: ReLU
    tiny_infer::ReLU relu;

    // 前向传播
    auto x1 = fc1.forward(input);
    auto output = relu.forward(x1);

    // 验证形状
    EXPECT_EQ(output.shape()[0], 4);
    EXPECT_EQ(output.shape()[1], 64);

    // 验证数值非负 (ReLU 特性)
    for(size_t i=0; i<output.size(); ++i) {
        EXPECT_GE(output.data()[i], 0.0f);
    }
}