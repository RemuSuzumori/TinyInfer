#pragma once
#include "tiny_infer/thread_pool.hpp"
#include <memory>
#include <mutex>

namespace tiny_infer {

// 单例模式管理全局资源
class Context {
public:
    // 获取单例实例 (C++11 之后的静态局部变量是线程安全的)
    static Context& instance() {
        static Context ctx;
        return ctx;
    }

    // 获取线程池
    ThreadPool& thread_pool() {
        return m_pool;
    }

    // 禁止拷贝
    Context(const Context&) = delete;
    Context& operator=(const Context&) = delete;

private:
    Context() : m_pool(std::thread::hardware_concurrency()) {} // 默认使用所有核心
    
    ThreadPool m_pool;
};

} // namespace tiny_infer