//
// demo_server —— 演示 Web 的后端可执行：注册模型族代表模型（det/cls 起步）并开 ServingServer。
//
// 核对的真实符号（Step 5b，记录于本文件顶部，均与仓库实际一致）：
//  - Det   类：modeldeploy::vision::detection::UltralyticsDet（csrc/vision/detection/ultralytics_det.h）
//            构造 (const std::string& model_file, const RuntimeOption&)；
//            predict -> std::vector<DetectionResult>*（多框）→ result_type = std::vector<DetectionResult>
//  - Cls   类：modeldeploy::vision::classification::Classification（csrc/vision/classification/classification.h）
//            构造 (const std::string& model_file, const RuntimeOption&)；
//            predict 缺 TimerArray 形参 → 用 ClassifyAdapter 桥接；result_type = ClassifyResult
//  - 设备  RuntimeOption::set_device(Device::CPU, id)（use_gpu 已废弃，见 runtime/runtime_option.h）
//  - 后端  RuntimeOption::use_ort_backend()
// 模型文件路径：repo/{name}/{ver}/model.onnx（构造器收 model_file，非目录）。
//
// 加载语义：缺权重 / 无设备 / 初始化失败 → 该模型注册为 ready=false（empty infer），
// 绝不因单个模型崩溃进程。det/cls 之外的未知族名给一个不可用占位。
#include <csignal>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <string>
#include <thread>

#include <nlohmann/json.hpp>

#include "serving/config.h"
#include "serving/model_repo.h"
#include "serving/model_entry.h"
#include "serving/adapters.h"
#include "serving/server.h"

#include "runtime/runtime_option.h"
#include "vision/common/result_json.h"
#include "vision/detection/ultralytics_det.h"
#include "vision/classification/classification.h"

using namespace modeldeploy;
using namespace modeldeploy::serving;

namespace fs = std::filesystem;

// 类别名表（按族）。模型目录下 labels.txt（每行一个）优先覆盖；这里给内置默认。
static const std::map<std::string, std::vector<std::string>>& builtin_labels() {
    static const std::map<std::string, std::vector<std::string>> m = {
        {"det", {"person", "bicycle", "car", "motorcycle", "airplane", "bus",
                 "train", "truck", "boat", "traffic light"}},
        {"cls", {"class0", "class1"}},
    };
    return m;
}

// 从模型目录读 labels.txt（可选）；无则用内置表。
static std::vector<std::string> read_labels(const std::string& dir, const std::string& type) {
    std::ifstream f(fs::path(dir) / "labels.txt");
    std::vector<std::string> labs;
    std::string line;
    while (std::getline(f, line)) {
        if (!line.empty()) labs.push_back(line);
    }
    if (!labs.empty()) return labs;
    auto it = builtin_labels().find(type);
    if (it != builtin_labels().end()) return it->second;
    return {};
}

// 构造一个 not-ready 的占位句柄（缺权重/设备/初始化失败的统一兜底）。
// 注意：infer 必须留空（default std::function）——model_repo.cpp 以
// `h.ready = (h.infer ? true : false)` 判定就绪，非空 infer 会把 ready 翻转为 true。
static ModelHandle not_ready_handle(const std::string& name, const std::string& ver) {
    ModelHandle h;
    h.name = name;
    h.version = ver;
    h.ready = false;
    return h;
}

// 按名构造真实模型句柄（每族一个 branch）。dir = repo/{name}/{ver}，含 model.onnx。
// 成功 → make_model_handle 已启动 AsyncModel；失败 → not-ready（不抛、不崩）。
static ModelHandle build_demo_handle(const std::string& name, const std::string& ver,
                                     const std::string& dir) {
    RuntimeOption opt;
    opt.use_ort_backend();
    opt.set_device(Device::CPU, 0);  // CPU 起步；GPU 演示可在有设备/权重时改 Device::GPU
    const fs::path model_file = fs::path(dir) / "model.onnx";

    if (name == "det") {
        try {
            using M = ResultModel<vision::detection::UltralyticsDet,
                                  std::vector<vision::DetectionResult>>;
            auto m = std::make_unique<M>(model_file.string(), opt);
            if (!m->is_initialized()) return not_ready_handle(name, ver);  // 缺权重/设备
            return make_model_handle(name, std::move(m));
        } catch (...) {
            return not_ready_handle(name, ver);
        }
    }
    if (name == "cls") {
        try {
            // classification 的 predict/batch_predict 缺 TimerArray → ClassifyAdapter 桥接
            using M = ClassifyAdapter<vision::ClassifyResult>;
            auto m = std::make_unique<M>(model_file.string(), opt);
            if (!m->is_initialized()) return not_ready_handle(name, ver);
            return make_model_handle(name, std::move(m));
        } catch (...) {
            return not_ready_handle(name, ver);
        }
    }
    // 其余族（seg/pose/ocr/face/lpr/obb/sem/depth）在后续任务按同一模式逐个补 branch。

    // 未知 name：给一个不可用占位，避免 scan 崩溃。
    return not_ready_handle(name, ver);
}

static void fill_meta(const std::string& dir, const std::string& type, ModelHandle* h) {
    h->type = type;
    h->labels = read_labels(dir, type);
    h->input_size = {640, 640};
}

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
            std::cout << "usage: demo_server [--port <port>] [--web <dir>] [--repo <dir>]\n";
            return 0;
        }
    }
    if (web_root.empty()) web_root = "web_demo";  // 构建目录下的 web 资产
    if (repo.empty()) repo = "demo_repo";

    ServingConfig cfg;
    cfg.host = "0.0.0.0";
    cfg.port = port;
    cfg.web_root = web_root;
    cfg.model_repo = repo;

    HandleBuilder builder = [](const std::string& name, const std::string& ver,
                               const std::string& dir) {
        ModelHandle h = build_demo_handle(name, ver, dir);
        fill_meta(dir, name, &h);  // type 用 name（det/cls），h.type 此时可能为空
        return h;
    };

    std::string err;
    ServingServer srv(cfg, std::move(builder), &err);
    if (!srv.start(&err)) {
        std::cerr << "start failed: " << err << "\n";
        return 1;
    }
    std::cout << "Demo serving on http://127.0.0.1:" << srv.port() << "/\n";
    std::signal(SIGINT, [](int) { /* 主线程忙等；Ctrl+C 结束 */ });
    while (true) std::this_thread::sleep_for(std::chrono::hours(1));
}
