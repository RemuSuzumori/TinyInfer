#pragma once

#include <array>
#include <cstdint>
#include <string>
#include <vector>

namespace tiny_infer::serving {

constexpr std::uint32_t kMagic = 0x53464954;  // "TIFS"
constexpr std::uint16_t kVersion = 1;

enum class MsgType : std::uint16_t {
    InferRequest = 1,
    InferResponse = 2,
};

enum class StatusCode : std::uint16_t {
    Ok = 0,
    BadRequest = 1,
    Busy = 2,
    Internal = 3,
};

struct Request {
    std::uint64_t request_id{0};
    std::vector<std::uint8_t> payload;
};

struct Response {
    std::uint64_t request_id{0};
    StatusCode status{StatusCode::Ok};
    std::vector<std::uint8_t> result;
};

constexpr std::size_t kRequestHeaderSize = 20;
constexpr std::size_t kResponseHeaderSize = 22;

bool parse_request_header(const std::array<std::uint8_t, kRequestHeaderSize>& buf,
                          std::uint64_t& request_id,
                          std::uint32_t& payload_len,
                          std::string& error);

std::vector<std::uint8_t> encode_response(const Response& response);

}  // namespace tiny_infer::serving
