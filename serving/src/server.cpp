#include "server.hpp"

#include "session.hpp"
#include "tiny_infer/node.hpp"
#include "tiny_infer/tensor.hpp"

#include <arpa/inet.h>
#include <chrono>
#include <cstring>
#include <iostream>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>

namespace tiny_infer::serving {

class Server::MnistModel {
public:
    explicit MnistModel(const std::string& model_dir) : fc1_(784, 128), fc2_(128, 10) {
        fc1_.load_params(model_dir + "/fc1_weights.bin", model_dir + "/fc1_bias.bin");
        fc2_.load_params(model_dir + "/fc2_weights.bin", model_dir + "/fc2_bias.bin");
    }

    std::pair<std::uint32_t, float> infer_top1(const std::vector<std::uint8_t>& payload) {
        Tensor input({1, 784});
        std::memcpy(input.data(), payload.data(), payload.size());
        auto h1 = fc1_.forward(input);
        auto h2 = relu_.forward(h1);
        auto out = fc2_.forward(h2);

        std::uint32_t label = 0;
        float best = out.data()[0];
        for (std::uint32_t i = 1; i < 10; ++i) {
            if (out.data()[i] > best) {
                best = out.data()[i];
                label = i;
            }
        }
        return {label, best};
    }

private:
    Linear fc1_;
    ReLU relu_;
    Linear fc2_;
};

Server::Server(ServerOptions options) : options_(std::move(options)), queue_(options_.queue_capacity) {
    if (options_.use_real_mnist) {
        mnist_model_ = std::make_unique<MnistModel>(options_.model_dir);
    }
}

Server::~Server() { stop(); }

void Server::run() {
    listen_fd_ = ::socket(AF_INET, SOCK_STREAM, 0);
    if (listen_fd_ < 0) throw std::runtime_error("socket failed");

    int enable = 1;
    setsockopt(listen_fd_, SOL_SOCKET, SO_REUSEADDR, &enable, sizeof(enable));

    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = INADDR_ANY;
    addr.sin_port = htons(options_.port);

    if (::bind(listen_fd_, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) < 0) throw std::runtime_error("bind failed");
    if (::listen(listen_fd_, 1024) < 0) throw std::runtime_error("listen failed");

    start_compute_workers();
    metrics_thread_ = std::thread([this] { print_metrics_loop(); });
    accept_thread_ = std::thread([this] { accept_loop(); });
    accept_thread_.join();
}

void Server::stop() {
    if (stop_flag_.exchange(true)) return;
    if (listen_fd_ >= 0) {
        ::shutdown(listen_fd_, SHUT_RDWR);
        ::close(listen_fd_);
        listen_fd_ = -1;
    }
    queue_.stop();
    if (accept_thread_.joinable()) accept_thread_.join();
    for (auto& t : compute_threads_) if (t.joinable()) t.join();
    if (metrics_thread_.joinable()) metrics_thread_.join();
}

void Server::accept_loop() {
    while (!stop_flag_) {
        sockaddr_in caddr{};
        socklen_t len = sizeof(caddr);
        int fd = ::accept(listen_fd_, reinterpret_cast<sockaddr*>(&caddr), &len);
        if (fd < 0) {
            if (!stop_flag_) continue;
            break;
        }
        std::thread([fd, this] { std::make_shared<Session>(fd, *this)->run(); }).detach();
    }
}

void Server::start_compute_workers() {
    for (std::size_t i = 0; i < options_.compute_workers; ++i) {
        compute_threads_.emplace_back([this] {
            Task task;
            while (queue_.pop(task)) {
                auto begin = std::chrono::steady_clock::now();
                Response response{task.request_id, StatusCode::Ok, {}};
                try {
                    response.result = run_inference(task.payload);
                    metrics_.success.fetch_add(1);
                } catch (...) {
                    response.status = StatusCode::Internal;
                    const std::string e = "internal_error";
                    response.result.assign(e.begin(), e.end());
                    metrics_.internal.fetch_add(1);
                }
                auto us = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - begin).count();
                metrics_.latency_us_sum.fetch_add(static_cast<std::uint64_t>(us));
                task.respond(std::move(response));
            }
        });
    }
}

std::vector<std::uint8_t> Server::run_inference(const std::vector<std::uint8_t>& payload) {
    if (options_.use_real_mnist) {
        if (payload.size() != 784 * sizeof(float)) throw std::runtime_error("invalid mnist payload size");
        auto [label, score] = mnist_model_->infer_top1(payload);
        std::string msg = "label=" + std::to_string(label) + ",score=" + std::to_string(score);
        return {msg.begin(), msg.end()};
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(5));
    std::string msg = "ok len=" + std::to_string(payload.size());
    return {msg.begin(), msg.end()};
}

void Server::print_metrics_loop() {
    using namespace std::chrono_literals;
    while (!stop_flag_) {
        std::this_thread::sleep_for(2s);
        auto succ = metrics_.success.load();
        auto avg = succ ? metrics_.latency_us_sum.load() / succ : 0;
        std::cout << "[metrics] total=" << metrics_.total_requests.load() << " success=" << succ
                  << " busy=" << metrics_.busy.load() << " bad=" << metrics_.bad_request.load()
                  << " internal=" << metrics_.internal.load() << " queue=" << queue_.size()
                  << " peak=" << queue_.peak_size() << " avg_us=" << avg << std::endl;
    }
}

}  // namespace tiny_infer::serving
