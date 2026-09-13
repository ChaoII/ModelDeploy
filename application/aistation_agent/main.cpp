#include <atomic>
#include <chrono>
#include <csignal>
#include <cstring>
#include <iostream>
#include <string>
#include <thread>
#include "agent_runtime.hpp"

static std::atomic<bool> g_stop{false};
static void on_signal(int) { g_stop = true; }

int main(int argc, char** argv) {
    AgentOptions o;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto next = [&](std::string* out) { if (i + 1 < argc) *out = argv[++i]; };
        if (a == "--host") next(&o.host);
        else if (a == "--port") o.port = std::stoi(argv[++i]);
        else if (a == "--data-dir") next(&o.data_dir);
        else if (a == "--api-key") next(&o.api_key);
        else if (a == "--cloud-url") next(&o.cloud_url);
        else if (a == "--edge-code") next(&o.edge_code);
        else if (a == "--secret") next(&o.secret);
        else if (a == "--model-cache-dir") next(&o.model_cache_dir);
        else if (a == "--s3-endpoint") next(&o.s3_endpoint);
        else if (a == "--max-channels") o.max_channels = std::stoi(argv[++i]);
        else if (a == "--heartbeat-interval") o.heartbeat_interval_sec = std::stoi(argv[++i]);
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
