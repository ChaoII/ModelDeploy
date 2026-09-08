//
// demo_server —— 演示 Web 的后端可执行：按手写 manifest 提供目录元数据，并开 ServingServer。
//
// 懒加载语义：启动只读 manifest 目录（status=Unloaded、infer 空），不实例化任何模型；
// 单槽实例化经 ModelRepo::load 触达（前端切模型时懒加载）。Task 4 在此按 type 各族构造真实
// SDK 模型；当前 HandleBuilder 仅回填清单元数据（infer 空 ⇒ load 置 Failed），供目录与生命周期跑通。
#include <chrono>
#include <csignal>
#include <cstdlib>
#include <iostream>
#include <string>
#include <thread>

#include "serving/config.h"
#include "serving/model_repo.h"
#include "serving/manifest.h"
#include "serving/server.h"

using namespace modeldeploy::serving;

int main(int argc, char** argv) {
    int port = 8000;
    std::string web_root, repo;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto val = [&]() -> std::string { return (i + 1 < argc) ? argv[++i] : ""; };
        if (a == "--port") port = std::stoi(val());
        else if (a == "--web") web_root = val();
        else if (a == "--repo") repo = val();
        else if (a == "--help") {
            std::cout << "usage: demo_server [--port <port>] [--web <dir>] [--repo <manifest.json>]\n";
            return 0;
        }
    }
    if (web_root.empty()) web_root = "web_demo";  // 构建目录下的 web 资产
    if (repo.empty()) repo = "demo_manifest.json";

    ServingConfig cfg;
    cfg.host = "0.0.0.0";
    cfg.port = port;
    cfg.web_root = web_root;
    cfg.model_repo = repo;

    HandleBuilder builder = [](const ManifestModel& m, const std::string&) {
        ModelHandle h;
        h.name = m.id;
        h.display = m.display;
        h.version = "1";
        h.type = m.type;
        h.input_size = m.input_size;
        h.labels = m.labels;
        return h;
    };

    std::string err;
    ServingServer srv(cfg, std::move(builder), &err);
    if (!srv.start(&err)) {
        std::cerr << "start failed: " << err << "\n";
        return 1;
    }
    std::cout << "Demo serving on http://127.0.0.1:" << srv.port() << "/\n";
    static volatile std::sig_atomic_t stop = 0;
    std::signal(SIGINT, [](int) { stop = 1; });
    std::signal(SIGTERM, [](int) { stop = 1; });
    while (!stop) std::this_thread::sleep_for(std::chrono::milliseconds(200));
    return 0;
}
