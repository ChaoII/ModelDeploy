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

int main(int argc, char* argv[]) {
    std::signal(SIGINT, handle_signal);
    std::signal(SIGTERM, handle_signal);

    int port = 8080;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--port" && i + 1 < argc) {
            port = std::atoi(argv[++i]);
        } else if (arg == "--help" || arg == "-h") {
            std::cerr << "Usage: " << argv[0] << " [--port PORT]" << std::endl;
            return 0;
        }
    }

    PipelineManager mgr;
    g_mgr = &mgr;

    HttpServer server(mgr, "0.0.0.0", port);
    if (!server.start()) {
        std::cerr << "[Main] Failed to start HTTP server" << std::endl;
        return 1;
    }

    std::cout << "[Main] Surveillance platform running on port " << port << ". Press Ctrl+C to stop." << std::endl;

    while (true) {
        std::this_thread::sleep_for(std::chrono::seconds(1));
    }

    return 0;
}
