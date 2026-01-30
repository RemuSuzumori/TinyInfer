#pragma once
#include <vector>
#include <numeric>
#include <stdexcept>
#include <algorithm> // std::fill

namespace tiny_infer {

class Tensor {
public:
    // 构造函数：传入形状 (例如 {2, 3} 代表 2行3列)
    Tensor(std::vector<size_t> shape) : m_shape(std::move(shape)) {
        if (m_shape.empty()) {
            throw std::runtime_error("Tensor shape cannot be empty");
        }
        // 计算总元素数量
        size_t total_size = 1;
        for (auto dim : m_shape) {
            if (dim == 0) throw std::runtime_error("Tensor dimension cannot be 0");
            total_size *= dim;
        }
        m_data.resize(total_size, 0.0f);
    }

    // 获取形状
    const std::vector<size_t>& shape() const { return m_shape; }

    // 元素总数（num elements）
    size_t numel() const { return m_data.size(); }

    // 兼容很多人习惯：size() == numel()
    size_t size() const { return m_data.size(); }

    // 获取原始数据指针
    float* data() { return m_data.data(); }
    const float* data() const { return m_data.data(); }

    // 方便调试：填充固定值
    void fill(float value) {
        std::fill(m_data.begin(), m_data.end(), value);
    }

    // 获取特定位置的元素 (带边界检查)
    float& at(size_t row, size_t col) {
        if (m_shape.size() != 2) throw std::runtime_error("Tensor is not 2D");
        return m_data[row * m_shape[1] + col];
    }

    const float& at(size_t row, size_t col) const {
        if (m_shape.size() != 2) throw std::runtime_error("Tensor is not 2D");
        return m_data[row * m_shape[1] + col];
    }

    // 声明矩阵乘法函数
    static Tensor matmul(const Tensor& a, const Tensor& b);

private:
    std::vector<size_t> m_shape;
    std::vector<float> m_data;
};

} // namespace tiny_infer
