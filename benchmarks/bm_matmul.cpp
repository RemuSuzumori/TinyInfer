#include <benchmark/benchmark.h>
#include "tiny_infer/tensor.hpp"
#include <random>

// 辅助函数：生成随机 Tensor，防止编译器优化掉计算过程
tiny_infer::Tensor create_random_tensor(const std::vector<size_t>& shape) {
    tiny_infer::Tensor t(shape);
    // 简单的随机填充
    for (size_t i = 0; i < t.shape()[0] * t.shape()[1]; ++i) {
        t.data()[i] = static_cast<float>(rand()) / RAND_MAX;
    }
    return t;
}

// 核心测试函数
static void BM_MatMul_Naive(benchmark::State& state) {
    // state.range(0) 获取我们在 main 中传入的矩阵大小 (例如 64, 128, 256...)
    long dim = state.range(0);
    
    // 1. 准备数据 (放在循环外，因为我们不测数据生成的耗时)
    tiny_infer::Tensor a = create_random_tensor({(size_t)dim, (size_t)dim});
    tiny_infer::Tensor b = create_random_tensor({(size_t)dim, (size_t)dim});

    // 2. 性能测试循环
    for (auto _ : state) {
        // 这里的代码会被重复运行多次以取平均值
        auto c = tiny_infer::Tensor::matmul(a, b);
        
        // 防止编译器把结果优化掉 (比如它发现 c 没被使用，可能就不算了)
        benchmark::DoNotOptimize(c);
    }

    // 3. 计算复杂度指标 (面试加分项)
    // 矩阵乘法 FLOPs = 2 * M * N * K
    double flops = 2.0 * dim * dim * dim;
    
    // 告诉 Benchmark 我们处理了多少个 Item (便于计算吞吐量)
    state.counters["GFLOPS"] = benchmark::Counter(
        flops, 
        benchmark::Counter::kIsIterationInvariantRate, // 自动除以运行时间
        benchmark::Counter::kIs1000 // 单位是 1000 (kilo/mega/giga)
    );
}

// 注册测试用例，并定义参数范围
// 从 64x64 测到 512x512
BENCHMARK(BM_MatMul_Naive)
    ->Arg(64)
    ->Arg(128)
    ->Arg(256)
    ->Arg(512)
    ->Unit(benchmark::kMillisecond); // 输出单位设为毫秒

BENCHMARK_MAIN();