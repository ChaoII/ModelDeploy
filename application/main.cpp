#include "pipeline_manager.hpp"
#include "http_server.hpp"
#include <iostream>
#include <csignal>
#include <cstdlib>
#include <thread>
#include <cuda_runtime.h>
#include <filesystem>

#ifdef _WIN32
#include <windows.h>
#endif

namespace fs = std::filesystem;

static PipelineManager* g_mgr = nullptr;
static std::string g_data_dir;
static bool g_running = true;

static std::string get_default_data_dir() {
#ifdef _WIN32
    char buf[MAX_PATH];
    if (GetModuleFileNameA(nullptr, buf, MAX_PATH) > 0) {
        std::string path(buf);
        auto pos = path.find_last_of("\\/");
        if (pos != std::string::npos) {
            return path.substr(0, pos) + "\\data";
        }
    }
#endif
    return "./data";
}

static void save_state() {
    if (g_mgr && !g_data_dir.empty()) {
        g_mgr->save_to_directory(g_data_dir);
    }
}

static void handle_signal(int sig) {
    std::cout << "\n[Main] Signal " << sig << ", shutting down..." << std::endl;
    g_running = false;
    // 先保存，再停止（stop_all 会清空 pipelines_）
    save_state();
    if (g_mgr) g_mgr->stop_all();
    std::exit(0);
}

int main(int argc, char* argv[]) {
    std::signal(SIGINT, handle_signal);
    std::signal(SIGTERM, handle_signal);

    int port = 18080;
    g_data_dir = get_default_data_dir();

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--port" && i + 1 < argc) {
            port = std::atoi(argv[++i]);
        } else if (arg == "--data-dir" && i + 1 < argc) {
            g_data_dir = argv[++i];
        } else if (arg == "--help" || arg == "-h") {
            std::cerr << "Usage: " << argv[0] << " [--port PORT] [--data-dir DIR]" << std::endl;
            return 0;
        }
    }

    cudaSetDevice(0);

    PipelineManager mgr;
    g_mgr = &mgr;

    // 加载持久化数据
    mgr.load_from_directory(g_data_dir);

    HttpServer server(mgr, "0.0.0.0", port);
    if (!server.start()) {
        std::cerr << "[Main] Failed to start HTTP server" << std::endl;
        return 1;
    }

    std::cout << "[Main] Running on port " << port << ", data dir: " << g_data_dir << std::endl;

    while (g_running) {
        std::this_thread::sleep_for(std::chrono::seconds(5));
        if (mgr.is_dirty()) {
            save_state();
        }
    }

    save_state();
    return 0;
}
