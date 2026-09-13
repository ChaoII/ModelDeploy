#include <atomic>
#include <chrono>
#include <csignal>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>
#include <thread>
#include "agent_runtime.hpp"

static std::atomic<bool> g_stop{false};
static void on_signal(int) { g_stop = true; }

// 解析整型参数：全程检查尾部消费量，非数字/越界/带多余字符一律失败（不抛到 main 外）。
static bool parse_int(const char* s, int* out, const std::string& flag) {
    try {
        size_t pos = 0;
        const int v = std::stoi(s, &pos);
        if (pos != std::strlen(s)) {
            std::cerr << "invalid value for " << flag << ": '" << s << "'" << std::endl;
            return false;
        }
        *out = v;
        return true;
    } catch (const std::exception& e) {
        std::cerr << "invalid value for " << flag << ": '" << s << "' (" << e.what() << ")"
                  << std::endl;
        return false;
    }
}

int main(int argc, char** argv) {
    AgentOptions o;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto next = [&](std::string* out) { if (i + 1 < argc) *out = argv[++i]; };
        auto int_arg = [&](int* out) {
            if (i + 1 >= argc) {
                std::cerr << "missing value for " << a << std::endl;
                return false;
            }
            return parse_int(argv[++i], out, a);
        };
        if (a == "--host") next(&o.host);
        else if (a == "--port") { if (!int_arg(&o.port)) return 1; }
        else if (a == "--data-dir") next(&o.data_dir);
        else if (a == "--api-key") next(&o.api_key);
        else if (a == "--cloud-url") next(&o.cloud_url);
        else if (a == "--edge-code") next(&o.edge_code);
        else if (a == "--secret") next(&o.secret);
        else if (a == "--model-cache-dir") next(&o.model_cache_dir);
        else if (a == "--s3-endpoint") next(&o.s3_endpoint);
        else if (a == "--max-channels") { if (!int_arg(&o.max_channels)) return 1; }
        else if (a == "--heartbeat-interval") { if (!int_arg(&o.heartbeat_interval_sec)) return 1; }
    }
    std::signal(SIGINT, on_signal);
    std::signal(SIGTERM, on_signal);

    AgentRuntime runtime(o);
    if (!runtime.start()) {
        std::cerr << "AgentRuntime failed to start" << std::endl;
        return 1;
    }
    std::cout << "[aistation_agent] listening on " << o.host << ":" << o.port
              << " edge_code=" << o.edge_code << std::endl;
    while (!g_stop.load()) std::this_thread::sleep_for(std::chrono::milliseconds(200));
    runtime.stop();
    return 0;
}
