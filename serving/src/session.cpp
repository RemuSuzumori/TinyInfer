#include "session.hpp"

#include <sys/socket.h>
#include <sys/time.h>
#include <unistd.h>

#include <array>
#include <future>

namespace tiny_infer::serving {

Session::Session(int fd, Server& server) : fd_(fd), server_(server) {
    timeval tv{};
    tv.tv_sec = static_cast<time_t>(server_.options().read_timeout_ms / 1000);
    tv.tv_usec = static_cast<suseconds_t>((server_.options().read_timeout_ms % 1000) * 1000);
    setsockopt(fd_, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));
}

Session::~Session() { close_fd(); }

bool Session::read_exact(std::uint8_t* dst, std::size_t len) {
    std::size_t got = 0;
    while (got < len && !closed_) {
        const auto r = ::recv(fd_, dst + got, len - got, 0);
        if (r <= 0) return false;
        got += static_cast<std::size_t>(r);
    }
    return got == len;
}

bool Session::write_exact(const std::uint8_t* src, std::size_t len) {
    std::size_t sent = 0;
    while (sent < len && !closed_) {
        const auto w = ::send(fd_, src + sent, len - sent, 0);
        if (w <= 0) return false;
        sent += static_cast<std::size_t>(w);
    }
    return sent == len;
}

void Session::run() {
    std::size_t inflight = 0;
    while (!closed_) {
        std::array<std::uint8_t, kRequestHeaderSize> header{};
        if (!read_exact(header.data(), header.size())) break;

        std::uint64_t request_id = 0;
        std::uint32_t payload_len = 0;
        std::string error;
        if (!parse_request_header(header, request_id, payload_len, error)) {
            server_.metrics().bad_request.fetch_add(1);
            auto out = encode_response(Response{request_id, StatusCode::BadRequest, {error.begin(), error.end()}});
            if (!write_exact(out.data(), out.size())) break;
            continue;
        }
        if (payload_len > server_.options().max_payload) {
            server_.metrics().bad_request.fetch_add(1);
            const std::string msg = "payload too large";
            auto out = encode_response(Response{request_id, StatusCode::BadRequest, {msg.begin(), msg.end()}});
            if (!write_exact(out.data(), out.size())) break;
            continue;
        }
        if (inflight >= server_.options().max_inflight_per_session) {
            server_.metrics().busy.fetch_add(1);
            const std::string msg = "too many in-flight";
            auto out = encode_response(Response{request_id, StatusCode::Busy, {msg.begin(), msg.end()}});
            if (!write_exact(out.data(), out.size())) break;
            continue;
        }

        std::vector<std::uint8_t> payload(payload_len);
        if (payload_len && !read_exact(payload.data(), payload.size())) break;

        server_.metrics().total_requests.fetch_add(1);
        auto promise = std::make_shared<std::promise<Response>>();
        auto fut = promise->get_future();

        Task task;
        task.request_id = request_id;
        task.payload = std::move(payload);
        task.respond = [promise](Response r) { promise->set_value(std::move(r)); };

        ++inflight;
        if (!server_.queue().try_push(std::move(task))) {
            --inflight;
            server_.metrics().busy.fetch_add(1);
            const std::string msg = "busy";
            auto out = encode_response(Response{request_id, StatusCode::Busy, {msg.begin(), msg.end()}});
            if (!write_exact(out.data(), out.size())) break;
            continue;
        }

        Response resp = fut.get();
        --inflight;
        auto out = encode_response(resp);
        if (!write_exact(out.data(), out.size())) break;
    }
    close_fd();
}

void Session::close_fd() {
    if (closed_.exchange(true)) return;
    if (fd_ >= 0) {
        ::shutdown(fd_, SHUT_RDWR);
        ::close(fd_);
        fd_ = -1;
    }
}

}  // namespace tiny_infer::serving
