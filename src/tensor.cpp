#include "tiny_infer/tensor.hpp"
#include "tiny_infer/context.hpp" // 引入上下文
#include <stdexcept>
#include <future>
#include <immintrin.h>
#include <vector>
#include <thread>
#include <algorithm>   // std::min

namespace tiny_infer {

    // 辅助函数：转置矩阵 (简单的单线程版本，因为只做一次)
    Tensor transpose(const Tensor& t) {
        size_t rows = t.shape()[0];
        size_t cols = t.shape()[1];
        Tensor result({cols, rows});
        
        for(size_t i=0; i<rows; ++i) {
            for(size_t j=0; j<cols; ++j) {
                result.at(j, i) = t.at(i, j);
            }
        }
        return result;
    }

    Tensor Tensor::matmul(const Tensor& a, const Tensor& b) {
        if (a.shape().size() != 2 || b.shape().size() != 2) throw std::runtime_error("2D only");
        size_t M = a.shape()[0];
        size_t K = a.shape()[1];
        size_t N = b.shape()[1];
        if (K != b.shape()[0]) throw std::runtime_error("Shape mismatch");

        Tensor c({M, N});
        
        // 1. 为了 SIMD 高效，先转置 B
        // 这样 B 的列在内存中就变成了连续的行
        Tensor b_t = transpose(b);

        const float* ptr_a = a.data();
        const float* ptr_b_t = b_t.data(); // 注意这里用转置后的数据
        float* ptr_c = c.data();

        auto& pool = Context::instance().thread_pool();
        size_t num_threads = std::thread::hardware_concurrency();
        size_t chunk_size = (M + num_threads - 1) / num_threads;

        std::vector<std::future<void>> futures;

        for (size_t t = 0; t < num_threads; ++t) {
            size_t start_row = t * chunk_size;
            size_t end_row = std::min(start_row + chunk_size, M);

            if (start_row >= end_row) break;

            futures.emplace_back(pool.enqueue([=] {
                for (size_t i = start_row; i < end_row; ++i) {
                    for (size_t j = 0; j < N; ++j) {
                        // 计算 A 的第 i 行 和 B_T 的第 j 行 的点积
                        __m256 sum_vec = _mm256_setzero_ps();
                        float sum_scalar = 0.0f;

                        size_t k = 0;
                        // AVX2 循环：一次算 8 个
                        for (; k + 8 <= K; k += 8) {
                            __m256 va = _mm256_loadu_ps(ptr_a + i * K + k);
                            __m256 vb = _mm256_loadu_ps(ptr_b_t + j * K + k); // 连续访问！
                            
                            // FMA: sum += va * vb
                            // _mm256_fmadd_ps(a, b, c) => a * b + c
                            sum_vec = _mm256_fmadd_ps(va, vb, sum_vec);
                        }

                        // 将 sum_vec 中的 8 个 float 累加起来
                        float temp[8];
                        _mm256_storeu_ps(temp, sum_vec);
                        for(int x=0; x<8; ++x) sum_scalar += temp[x];

                        // 处理剩余的尾巴 (K 不能被 8 整除的部分)
                        for (; k < K; ++k) {
                            sum_scalar += ptr_a[i * K + k] * ptr_b_t[j * K + k];
                        }

                        ptr_c[i * N + j] = sum_scalar;
                    }
                }
            }));
        }

        for (auto& f : futures) f.get();
        return c;
    }

}