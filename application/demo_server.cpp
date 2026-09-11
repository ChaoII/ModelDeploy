//
// demo_server —— 演示 Web 的后端可执行：按手写 manifest 提供目录元数据，并开 ServingServer。
//
// 懒加载语义：启动只读 manifest 目录（status=Unloaded、infer 空），不实例化任何模型；
// 单槽实例化经 ModelRepo::load 触达（前端切模型时懒加载）。模型构造复用 SDK 内置
// HandleBuilder（csrc/serving/builtin_builder.*），故此处不再内联各族构造逻辑。
//
#include <chrono>
#include <csignal>
#include <cstdlib>
#include <iostream>
#include <string>
#include <thread>

#include "serving/config.h"
#include "serving/server.h"

using namespace modeldeploy::serving;

int main(int argc, char** argv) {
    int port = 8000;
    std::string web_root, repo, font = "test_data/msyh.ttc";
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto val = [&]() -> std::string { return (i + 1 < argc) ? argv[++i] : ""; };
        if (a == "--port") port = std::stoi(val());
        else if (a == "--web") web_root = val();
        else if (a == "--repo") repo = val();
        else if (a == "--font") font = val();
        else if (a == "--help") {
            std::cout << "usage: demo_server [--port <port>] [--web <dir>] [--repo <manifest.json>]"
                         " [--font <ttf>]\n";
            std::cout << "  从仓库根目录运行：资源根由 manifest 的 base 字段决定。\n";
            std::cout << "  --repo  manifest.json 路径（缺省 application/demo_manifest.json）\n";
            std::cout << "  --web   静态 web 资产目录（缺省 web_demo）\n";
            std::cout << "  --font  可视化字体路径（缺省 test_data/msyh.ttc）\n";
            return 0;
        }
    }
    if (web_root.empty()) web_root = "web_demo";  // 构建目录下的 web 资产
    if (repo.empty()) repo = "application/demo_manifest.json";

    ServingConfig cfg;
    cfg.host = "0.0.0.0";
    cfg.port = port;
    cfg.web_root = web_root;
    cfg.model_repo = repo;
    cfg.font_path = font;

    std::string err;
    ServingServer srv(cfg, nullptr, &err);  // nullptr → 用 SDK 内置通用 builder
    if (!srv.start(&err)) {
        std::cerr << "start failed: " << err << "\n";
        return 1;
    }
    std::cout << "Demo serving on http://127.0.0.1:" << srv.port() << "/\n";
    // SIGINT/SIGTERM → 置位停止标志（仅 async-signal-safe 的 sig_atomic_t 赋值）。
    static volatile std::sig_atomic_t stop = 0;
    std::signal(SIGINT, [](int) { stop = 1; });
    std::signal(SIGTERM, [](int) { stop = 1; });
    while (!stop) std::this_thread::sleep_for(std::chrono::milliseconds(200));
    return 0;
}
