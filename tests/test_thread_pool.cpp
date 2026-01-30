#include <gtest/gtest.h>
#include "tiny_infer/thread_pool.hpp"
#include <atomic>

// 测试基本并行能力
TEST(ThreadPoolTest, SimpleParallel) {
    tiny_infer::ThreadPool pool(4);
    std::atomic<int> counter{0};

    std::vector<std::future<void>> results;
    for(int i = 0; i < 100; ++i) {
        results.emplace_back(
            pool.enqueue([&counter] {
                counter++;
            })
        );
    }

    // 等待所有任务完成
    for(auto && result: results)
        result.get();
        
    EXPECT_EQ(counter, 100);
}

// 测试能否正确获取返回值
TEST(ThreadPoolTest, ReturnValue) {
    tiny_infer::ThreadPool pool(2);
    
    auto f1 = pool.enqueue([](int a, int b){ return a + b; }, 10, 20);
    auto f2 = pool.enqueue([]{ return std::string("hello"); });

    EXPECT_EQ(f1.get(), 30);
    EXPECT_EQ(f2.get(), std::string("hello"));
}