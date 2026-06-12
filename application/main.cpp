#include <iostream>
#include <csignal>
#include <cstdlib>
#include <thread>

#include "pipeline_manager.hpp"
#include "http_server.hpp"

static PipelineManager* g_mgr = nullptr;

static void handle_signal(int sig) {
    std::cout << "\n[Main] Signal " << sig << ", shutting down..." << std::endl;
    if (g_mgr) g_mgr->stop_all();
    std::exit(0);
}

static void print_usage(const char* prog) {
    std::cerr << "Usage: " << prog << " [--port PORT] [--config FILE]" << std::endl;
    std::cerr << "  --port PORT     HTTP API port (default: 8080)" << std::endl;
    std::cerr << "  --config FILE   JSON config file with tasks to pre-load" << std::endl;
}

int main(int argc, char* argv[]) {
    std::signal(SIGINT, handle_signal);
    std::signal(SIGTERM, handle_signal);

    int port = 8080;
    std::string config_file;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--port" && i + 1 < argc) {
            port = std::atoi(argv[++i]);
        } else if (arg == "--config" && i + 1 < argc) {
            config_file = argv[++i];
        } else if (arg == "--help" || arg == "-h") {
            print_usage(argv[0]);
            return 0;
        } else {
            print_usage(argv[0]);
            return 1;
        }
    }

    PipelineManager mgr;
    g_mgr = &mgr;

    // Pre-load tasks from config file
    if (!config_file.empty()) {
        // TODO: implement config file loading
        std::cerr << "[Main] Config file loading not yet implemented" << std::endl;
    }

    HttpServer server(mgr, "0.0.0.0", port);
    if (!server.start()) {
        std::cerr << "[Main] Failed to start HTTP server" << std::endl;
        return 1;
    }

    std::cout << "[Main] Surveillance platform running. Press Ctrl+C to stop." << std::endl;

    // Block main thread
    while (true) {
        std::this_thread::sleep_for(std::chrono::seconds(1));
    }

    return 0;
}
