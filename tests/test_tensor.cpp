#include <gtest/gtest.h>
#include "tiny_infer/tensor.hpp"

// 测试初始化和形状
TEST(TensorTest, BasicShape) {
    tiny_infer::Tensor t({2, 3}); // 2x3 矩阵
    EXPECT_EQ(t.shape()[0], 2);
    EXPECT_EQ(t.shape()[1], 3);
    // 2*3 = 6 个元素，这里没有 size() 函数了，如果你想测可以自己算，或者在 header 加回 size()
    // 为了简单，我们只测 shape
}

// 测试简单的矩阵乘法
TEST(TensorTest, SimpleMatMul) {
    // A (2x2)
    tiny_infer::Tensor a({2, 2});
    a.at(0, 0) = 1.0f; a.at(0, 1) = 2.0f;
    a.at(1, 0) = 3.0f; a.at(1, 1) = 4.0f;

    // B (2x2) - 单位矩阵 * 0.5
    tiny_infer::Tensor b({2, 2});
    b.at(0, 0) = 0.5f; b.at(0, 1) = 0.0f;
    b.at(1, 0) = 0.0f; b.at(1, 1) = 0.5f;

    // C = A * B
    auto c = tiny_infer::Tensor::matmul(a, b);

    // 验证结果
    EXPECT_EQ(c.shape()[0], 2);
    EXPECT_EQ(c.shape()[1], 2);

    EXPECT_FLOAT_EQ(c.at(0, 0), 0.5f);
    EXPECT_FLOAT_EQ(c.at(0, 1), 1.0f);
    EXPECT_FLOAT_EQ(c.at(1, 0), 1.5f);
    EXPECT_FLOAT_EQ(c.at(1, 1), 2.0f);
}

// 测试异常情况
TEST(TensorTest, MatMulShapeMismatch) {
    tiny_infer::Tensor a({2, 3});
    tiny_infer::Tensor b({4, 5}); // 3 != 4
    
    EXPECT_THROW(tiny_infer::Tensor::matmul(a, b), std::runtime_error);
}