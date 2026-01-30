add_executable(run_benchmarks
    bm_matmul.cpp
)

target_link_libraries(run_benchmarks PRIVATE
    TinyInfer           # 测谁？测你的库
    benchmark::benchmark_main # Google Benchmark 的入口
)