#pragma once

#include "server.hpp"

#include <atomic>
#include <cstdint>
#include <memory>

namespace tiny_infer::serving {

class Session : public std::enable_shared_from_this<Session> {
public:
    Session(int fd, Server& server);
    ~Session();

    void run();

private:
    bool read_exact(std::uint8_t* dst, std::size_t len);
    bool write_exact(const std::uint8_t* src, std::size_t len);
    void close_fd();

    int fd_{-1};
    Server& server_;
    std::atomic<bool> closed_{false};
};

}  // namespace tiny_infer::serving
