#include "server.hpp"

#include <cstdlib>
#include <iostream>
#include <string>

using tiny_infer::serving::Server;
using tiny_infer::serving::ServerOptions;

namespace {

void print_help() {
    std::cout << "tinyinfer-serving options:\n"
              << "  --help\n"
              << "  --port <n>\n"
              << "  --io-threads <n>\n"
              << "  --compute-workers <n>\n"
              << "  --queue-capacity <n>\n"
              << "  --max-payload <bytes>\n"
              << "  --read-timeout-ms <n>\n"
              << "  --max-inflight <n>\n"
              << "  --real-mnist\n"
              << "  --model-dir <path>\n";
}

}  // namespace

int main(int argc, char** argv) {
    ServerOptions options;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        auto get_next = [&](const char* name) -> const char* {
            if (i + 1 >= argc) {
                std::cerr << "missing value for " << name << std::endl;
                std::exit(1);
            }
            return argv[++i];
        };

        if (arg == "--help") {
            print_help();
            return 0;
        }
        if (arg == "--port") options.port = static_cast<std::uint16_t>(std::stoi(get_next("--port")));
        else if (arg == "--io-threads") options.io_threads = static_cast<std::size_t>(std::stoul(get_next("--io-threads")));
        else if (arg == "--compute-workers") options.compute_workers = static_cast<std::size_t>(std::stoul(get_next("--compute-workers")));
        else if (arg == "--queue-capacity") options.queue_capacity = static_cast<std::size_t>(std::stoul(get_next("--queue-capacity")));
        else if (arg == "--max-payload") options.max_payload = static_cast<std::size_t>(std::stoul(get_next("--max-payload")));
        else if (arg == "--read-timeout-ms") options.read_timeout_ms = static_cast<std::size_t>(std::stoul(get_next("--read-timeout-ms")));
        else if (arg == "--max-inflight") options.max_inflight_per_session = static_cast<std::size_t>(std::stoul(get_next("--max-inflight")));
        else if (arg == "--real-mnist") options.use_real_mnist = true;
        else if (arg == "--model-dir") options.model_dir = get_next("--model-dir");
        else {
            std::cerr << "unknown arg: " << arg << std::endl;
            print_help();
            return 1;
        }
    }

    try {
        std::cout << "starting tinyinfer-serving on port " << options.port << std::endl;
        Server server(options);
        server.run();
    } catch (const std::exception& ex) {
        std::cerr << "fatal: " << ex.what() << std::endl;
        return 1;
    }

    return 0;
}
