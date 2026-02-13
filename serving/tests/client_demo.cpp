#include <arpa/inet.h>

#include <cstdint>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>

namespace {

void write_u16(std::vector<std::uint8_t>& b, std::uint16_t v) {
    b.push_back(v & 0xFF);
    b.push_back((v >> 8) & 0xFF);
}
void write_u32(std::vector<std::uint8_t>& b, std::uint32_t v) {
    for (int i = 0; i < 4; ++i) b.push_back((v >> (8 * i)) & 0xFF);
}
void write_u64(std::vector<std::uint8_t>& b, std::uint64_t v) {
    for (int i = 0; i < 8; ++i) b.push_back((v >> (8 * i)) & 0xFF);
}

std::uint16_t read_u16(const std::uint8_t* p) { return p[0] | (p[1] << 8); }
std::uint32_t read_u32(const std::uint8_t* p) { return p[0] | (p[1] << 8) | (p[2] << 16) | (p[3] << 24); }

bool read_exact(int fd, std::uint8_t* dst, std::size_t n) {
    std::size_t got = 0;
    while (got < n) {
        const auto r = ::recv(fd, dst + got, n - got, 0);
        if (r <= 0) return false;
        got += static_cast<std::size_t>(r);
    }
    return true;
}

}  // namespace

int main(int argc, char** argv) {
    std::string host = "127.0.0.1";
    int port = 9000;
    std::string payload = "hello";

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--host" && i + 1 < argc) host = argv[++i];
        else if (arg == "--port" && i + 1 < argc) port = std::stoi(argv[++i]);
        else if (arg == "--payload" && i + 1 < argc) payload = argv[++i];
    }

    const int fd = ::socket(AF_INET, SOCK_STREAM, 0);
    if (fd < 0) return 1;

    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_port = htons(port);
    inet_pton(AF_INET, host.c_str(), &addr.sin_addr);

    if (::connect(fd, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) != 0) {
        std::perror("connect");
        return 1;
    }

    std::vector<std::uint8_t> req;
    write_u32(req, 0x53464954);
    write_u16(req, 1);
    write_u16(req, 1);
    write_u64(req, 42);
    write_u32(req, static_cast<std::uint32_t>(payload.size()));
    req.insert(req.end(), payload.begin(), payload.end());

    if (::send(fd, req.data(), req.size(), 0) < 0) {
        std::perror("send");
        return 1;
    }

    std::vector<std::uint8_t> hdr(22);
    if (!read_exact(fd, hdr.data(), hdr.size())) return 1;

    const auto status = read_u16(hdr.data() + 16);
    const auto len = read_u32(hdr.data() + 18);
    std::vector<std::uint8_t> body(len);
    if (len && !read_exact(fd, body.data(), len)) return 1;

    std::cout << "status=" << status << " result=" << std::string(body.begin(), body.end()) << std::endl;
    return 0;
}
