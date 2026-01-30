#pragma once

#include <vector>
#include <queue>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <functional>
#include <future>
#include <atomic>
#include <type_traits>

namespace tiny_infer {

class ThreadPool {
public:
    // 构造函数：启动 num_threads 个工作线程
    explicit ThreadPool(size_t num_threads = std::thread::hardware_concurrency()) 
        : m_stop(false) 
    {
        for(size_t i = 0; i < num_threads; ++i) {
            // emplace_back 直接在 vector 尾部构造线程对象
            m_workers.emplace_back([this] {
                // 工作线程的主循环
                while(true) {
                    std::function<void()> task;

                    {
                        // 1. 获取锁
                        std::unique_lock<std::mutex> lock(this->m_queue_mutex);
                        
                        // 2. 等待条件满足：要么停止了，要么有任务了
                        // wait 会自动释放锁并阻塞，直到 notify 唤醒
                        this->m_condition.wait(lock, [this]{ 
                            return this->m_stop || !this->m_tasks.empty(); 
                        });

                        // 3. 如果停止且没任务，线程退出
                        if(this->m_stop && this->m_tasks.empty())
                            return;

                        // 4. 取出任务 (move 提高性能)
                        task = std::move(this->m_tasks.front());
                        this->m_tasks.pop();
                    }

                    // 5. 执行任务 (锁外执行，允许并发)
                    task();
                }
            });
        }
    }

    // 析构函数：优雅关闭
    ~ThreadPool() {
        {
            std::unique_lock<std::mutex> lock(m_queue_mutex);
            m_stop = true;
        }
        // 唤醒所有睡大觉的线程，让它们看到 stop 标志并退出
        m_condition.notify_all();
        
        // 等待所有线程执行完手头的活
        for(std::thread &worker : m_workers) {
            if(worker.joinable())
                worker.join();
        }
    }

    // 核心接口：提交任务
    // 使用可变参数模板 (Variadic Templates) 支持任意函数和参数
    // 返回 std::future，允许获取返回值
    template<class F, class... Args>
    auto enqueue(F&& f, Args&&... args) 
        -> std::future<typename std::invoke_result<F, Args...>::type>
    {
        using return_type = typename std::invoke_result<F, Args...>::type;

        // 将任务打包成 shared_ptr<packaged_task>，以便能复制进 lambda
        auto task = std::make_shared<std::packaged_task<return_type()>>(
            std::bind(std::forward<F>(f), std::forward<Args>(args)...)
        );
        
        std::future<return_type> res = task->get_future();
        {
            std::unique_lock<std::mutex> lock(m_queue_mutex);

            // 如果池子停了还提交，抛异常
            if(m_stop)
                throw std::runtime_error("enqueue on stopped ThreadPool");

            // 把任务放入队列
            // 这里的 lambda 只是为了适配 void() 签名，实际执行的是 packaged_task
            m_tasks.emplace([task](){ (*task)(); });
        }
        
        // 唤醒一个线程去干活
        m_condition.notify_one();
        return res;
    }

private:
    // 线程数组
    std::vector<std::thread> m_workers;
    // 任务队列
    std::queue<std::function<void()>> m_tasks;
    
    // 同步工具
    std::mutex m_queue_mutex;
    std::condition_variable m_condition;
    bool m_stop;
};

} // namespace tiny_infer