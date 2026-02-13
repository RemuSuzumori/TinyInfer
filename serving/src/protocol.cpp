#include "protocol.hpp"

namespace tiny_infer::serving {
namespace {

std::uint16_t read_u16_le(const std::uint8_t* p) {
    return static_cast<std::uint16_t>(p[0] | (p[1] << 8));
}

std::uint32_t read_u32_le(const std::uint8_t* p) {
    return static_cast<std::uint32_t>(p[0] | (p[1] << 8) | (p[2] << 16) | (p[3] << 24));
}

std::uint64_t read_u64_le(const std::uint8_t* p) {
    std::uint64_t v = 0;
    for (int i = 0; i < 8; ++i) {
        v |= (static_cast<std::uint64_t>(p[i]) << (i * 8));
    }
    return v;
}

void write_u16_le(std::vector<std::uint8_t>& out, std::uint16_t v) {
    out.push_back(static_cast<std::uint8_t>(v & 0xFF));
    out.push_back(static_cast<std::uint8_t>((v >> 8) & 0xFF));
}

void write_u32_le(std::vector<std::uint8_t>& out, std::uint32_t v) {
    for (int i = 0; i < 4; ++i) {
        out.push_back(static_cast<std::uint8_t>((v >> (i * 8)) & 0xFF));
    }
}

void write_u64_le(std::vector<std::uint8_t>& out, std::uint64_t v) {
    for (int i = 0; i < 8; ++i) {
        out.push_back(static_cast<std::uint8_t>((v >> (i * 8)) & 0xFF));
    }
}

}  // namespace

bool parse_request_header(const std::array<std::uint8_t, kRequestHeaderSize>& buf,
                          std::uint64_t& request_id,
                          std::uint32_t& payload_len,
                          std::string& error) {
    const auto magic = read_u32_le(buf.data());
    const auto version = read_u16_le(buf.data() + 4);
    const auto msg_type = read_u16_le(buf.data() + 6);

    if (magic != kMagic) {
        error = "bad magic";
        return false;
    }
    if (version != kVersion) {
        error = "bad version";
        return false;
    }
    if (msg_type != static_cast<std::uint16_t>(MsgType::InferRequest)) {
        error = "bad message type";
        return false;
    }

    request_id = read_u64_le(buf.data() + 8);
    payload_len = read_u32_le(buf.data() + 16);
    return true;
}

std::vector<std::uint8_t> encode_response(const Response& response) {
    std::vector<std::uint8_t> out;
    out.reserve(kResponseHeaderSize + response.result.size());

    write_u32_le(out, kMagic);
    write_u16_le(out, kVersion);
    write_u16_le(out, static_cast<std::uint16_t>(MsgType::InferResponse));
    write_u64_le(out, response.request_id);
    write_u16_le(out, static_cast<std::uint16_t>(response.status));
    write_u32_le(out, static_cast<std::uint32_t>(response.result.size()));

    out.insert(out.end(), response.result.begin(), response.result.end());
    return out;
}

}  // namespace tiny_infer::serving
