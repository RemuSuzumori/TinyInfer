#pragma once

#include "bounded_queue.hpp"
#include "protocol.hpp"

#include <atomic>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <thread>
#include <vector>

namespace tiny_infer::serving {

struct Task {
    std::uint64_t request_id{0};
    std::vector<std::uint8_t> payload;
    std::function<void(Response)> respond;
};

struct Metrics {
    std::atomic<std::uint64_t> total_requests{0};
    std::atomic<std::uint64_t> success{0};
    std::atomic<std::uint64_t> busy{0};
    std::atomic<std::uint64_t> bad_request{0};
    std::atomic<std::uint64_t> internal{0};
    std::atomic<std::uint64_t> latency_us_sum{0};
};

struct ServerOptions {
    std::uint16_t port{9000};
    std::size_t io_threads{1};
    std::size_t compute_workers{4};
    std::size_t queue_capacity{1024};
    std::size_t max_payload{1 << 20};
    std::size_t max_inflight_per_session{16};
    std::size_t read_timeout_ms{5000};
    bool use_real_mnist{false};
    std::string model_dir{"models"};
};

class Server {
public:
    explicit Server(ServerOptions options);
    ~Server();

    void run();
    void stop();

    BoundedQueue<Task>& queue() { return queue_; }
    Metrics& metrics() { return metrics_; }
    const ServerOptions& options() const { return options_; }

    std::vector<std::uint8_t> run_inference(const std::vector<std::uint8_t>& payload);

private:
    void accept_loop();
    void start_compute_workers();
    void print_metrics_loop();

    ServerOptions options_;
    int listen_fd_{-1};
    BoundedQueue<Task> queue_;
    Metrics metrics_;
    std::atomic<bool> stop_flag_{false};

    std::thread accept_thread_;
    std::vector<std::thread> compute_threads_;
    std::thread metrics_thread_;

    class MnistModel;
    std::unique_ptr<MnistModel> mnist_model_;
};

}  // namespace tiny_infer::serving
