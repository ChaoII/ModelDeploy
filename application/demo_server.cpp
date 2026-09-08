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
//  - Seg   类：modeldeploy::vision::detection::UltralyticsSeg（csrc/vision/iseg/ultralytics_seg.h）
//            predict -> std::vector<InstanceSegResult>*，batch -> vector<vector> → R=std::vector<InstanceSegResult>
//  - Pose  类：modeldeploy::vision::detection::UltralyticsPose（csrc/vision/pose/ultralytics_pose.h）
//            R=std::vector<KeyPointsResult>
//  - Obb   类：modeldeploy::vision::detection::UltralyticsObb（csrc/vision/obb/ultralytics_obb.h）
//            R=std::vector<ObbResult>
//  - Sem   类：modeldeploy::vision::detection::UltralyticsSem（csrc/vision/sem/ultralytics_sem.h）
//            predict -> SemSegResult*（单掩码）→ R=SemSegResult
//  - Depth 类：modeldeploy::vision::detection::UltralyticsDepth（csrc/vision/depth/ultralytics_depth.h）
//            R=DepthResult
//  - Ocr   类：modeldeploy::vision::ocr::PaddleOCR（csrc/vision/ocr/ppocr.h）
//            构造 (det_path, cls_path, rec_path, dict_path, RuntimeOption)；dir 内 model.onnx(det)/rec.onnx/cls.onnx(可选)/dict.txt
//            predict -> OCRResult*（单）→ R=OCRResult
//  - Face  类：modeldeploy::vision::face::Scrfd（csrc/vision/face/face_det/scrfd.h）
//            predict -> std::vector<KeyPointsResult>*（人脸框+5关键点）→ R=std::vector<KeyPointsResult>
//  - Lpr   类：modeldeploy::vision::lpr::LprPipeline（csrc/vision/lpr/lpr_pipeline/lpr_pipeline.h）
//            构造 (det_path, rec_path, RuntimeOption)；predict -> std::vector<LprResult>*
//            无 batch_predict → 用 LprAdapter 补 batch；R=std::vector<LprResult>
//  - 设备  RuntimeOption::set_device(Device::CPU, id)（use_gpu 已废弃，见 runtime/runtime_option.h）
//  - 后端  RuntimeOption::use_ort_backend()
// 模型文件路径：repo/{name}/{ver}/model.onnx（构造器收 model_file，非目录）；
//              ocr/lpr 为多文件族：model.onnx 作 det/主模型，另附 rec.onnx(+cls.onnx/dict.txt)。
//
// 加载语义：缺权重 / 无设备 / 初始化失败 → 该模型注册为 ready=false（empty infer），
// 绝不因单个模型崩溃进程。未知族名给一个不可用占位。
#include <chrono>
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
#include "vision/iseg/ultralytics_seg.h"
#include "vision/pose/ultralytics_pose.h"
#include "vision/ocr/ppocr.h"
#include "vision/face/face_det/scrfd.h"
#include "vision/lpr/lpr_pipeline/lpr_pipeline.h"
#include "vision/obb/ultralytics_obb.h"
#include "vision/sem/ultralytics_sem.h"
#include "vision/depth/ultralytics_depth.h"

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

// 目录名（族名）→ 前端渲染器 type。repo 扫描用目录名作模型名，各族名即 type；
// 归一为显式映射，未知目录名沿用自身。
static const std::map<std::string, std::string>& family_type_map() {
    static const std::map<std::string, std::string> m = {
        {"det", "det"},   {"cls", "cls"},   {"seg", "seg"}, {"pose", "pose"},
        {"ocr", "ocr"},   {"face", "face"}, {"lpr", "lpr"}, {"obb", "obb"},
        {"sem", "sem"},   {"depth", "depth"},
    };
    return m;
}

// 族名 → 输入尺寸 {w,h}（前端侧边栏展示 / 占位元数据）。实际推理由各模型预处理器
// 根据权重决定；此处仅作 UI 元数据，缺值回退 640x640。
static const std::map<std::string, std::vector<int>>& family_input_map() {
    static const std::map<std::string, std::vector<int>> m = {
        {"det", {640, 640}}, {"cls", {640, 640}}, {"seg", {640, 640}},
        {"pose", {640, 640}}, {"ocr", {640, 640}}, {"face", {640, 640}},
        {"lpr", {640, 640}}, {"obb", {640, 640}}, {"sem", {640, 640}},
        {"depth", {640, 640}},
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
    if (name == "seg") {
        try {
            using M = ResultModel<vision::detection::UltralyticsSeg,
                                  std::vector<vision::InstanceSegResult>>;
            auto m = std::make_unique<M>(model_file.string(), opt);
            if (!m->is_initialized()) return not_ready_handle(name, ver);
            return make_model_handle(name, std::move(m));
        } catch (...) {
            return not_ready_handle(name, ver);
        }
    }
    if (name == "pose") {
        try {
            using M = ResultModel<vision::detection::UltralyticsPose,
                                  std::vector<vision::KeyPointsResult>>;
            auto m = std::make_unique<M>(model_file.string(), opt);
            if (!m->is_initialized()) return not_ready_handle(name, ver);
            return make_model_handle(name, std::move(m));
        } catch (...) {
            return not_ready_handle(name, ver);
        }
    }
    if (name == "obb") {
        try {
            using M = ResultModel<vision::detection::UltralyticsObb,
                                  std::vector<vision::ObbResult>>;
            auto m = std::make_unique<M>(model_file.string(), opt);
            if (!m->is_initialized()) return not_ready_handle(name, ver);
            return make_model_handle(name, std::move(m));
        } catch (...) {
            return not_ready_handle(name, ver);
        }
    }
    if (name == "sem") {
        try {
            using M = ResultModel<vision::detection::UltralyticsSem, vision::SemSegResult>;
            auto m = std::make_unique<M>(model_file.string(), opt);
            if (!m->is_initialized()) return not_ready_handle(name, ver);
            return make_model_handle(name, std::move(m));
        } catch (...) {
            return not_ready_handle(name, ver);
        }
    }
    if (name == "depth") {
        try {
            using M = ResultModel<vision::detection::UltralyticsDepth, vision::DepthResult>;
            auto m = std::make_unique<M>(model_file.string(), opt);
            if (!m->is_initialized()) return not_ready_handle(name, ver);
            return make_model_handle(name, std::move(m));
        } catch (...) {
            return not_ready_handle(name, ver);
        }
    }
    if (name == "face") {
        try {
            // Scrfd：人脸检测，predict 输出人脸框 + 5 关键点 → vector<KeyPointsResult>
            using M = ResultModel<vision::face::Scrfd, std::vector<vision::KeyPointsResult>>;
            auto m = std::make_unique<M>(model_file.string(), opt);
            if (!m->is_initialized()) return not_ready_handle(name, ver);
            return make_model_handle(name, std::move(m));
        } catch (...) {
            return not_ready_handle(name, ver);
        }
    }
    if (name == "ocr") {
        try {
            // PaddleOCR 构造收 (det, cls, rec, dict, option)；dir 约定 model.onnx(det)+rec.onnx
            // +cls.onnx(可选)+dict.txt；cls 缺失传空串（内部自动禁用方向分类）。
            const fs::path rec = fs::path(dir) / "rec.onnx";
            const fs::path dict = fs::path(dir) / "dict.txt";
            const fs::path cls = fs::path(dir) / "cls.onnx";
            std::string cls_path = fs::exists(cls) ? cls.string() : "";
            using M = ResultModel<vision::ocr::PaddleOCR, vision::OCRResult>;
            auto m = std::make_unique<M>(model_file.string(), cls_path, rec.string(),
                                         dict.string(), opt);
            if (!m->is_initialized()) return not_ready_handle(name, ver);
            return make_model_handle(name, std::move(m));
        } catch (...) {
            return not_ready_handle(name, ver);
        }
    }
    if (name == "lpr") {
        try {
            // LprPipeline 构造收 (det, rec, option)；dir 约定 model.onnx(det)+rec.onnx；
            // 无 batch_predict → LprAdapter 补 batch。
            const fs::path rec = fs::path(dir) / "rec.onnx";
            using M = LprAdapter<std::vector<vision::LprResult>>;
            auto m = std::make_unique<M>(model_file.string(), rec.string(), opt);
            if (!m->is_initialized()) return not_ready_handle(name, ver);
            return make_model_handle(name, std::move(m));
        } catch (...) {
            return not_ready_handle(name, ver);
        }
    }
    // 其余族 / 未知 name：给不可用占位，避免 scan 崩溃。
    return not_ready_handle(name, ver);
}

static void fill_meta(const std::string& dir, const std::string& name, ModelHandle* h) {
    auto it_type = family_type_map().find(name);
    h->type = it_type != family_type_map().end() ? it_type->second : name;
    auto it_size = family_input_map().find(name);
    h->input_size = it_size != family_input_map().end() ? it_size->second
                                                        : std::vector<int>{640, 640};
    h->labels = read_labels(dir, h->type);
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
        fill_meta(dir, name, &h);  // type/input_size 由族名映射填充
        return h;
    };

    std::string err;
    ServingServer srv(cfg, std::move(builder), &err);
    if (!srv.start(&err)) {
        std::cerr << "start failed: " << err << "\n";
        return 1;
    }
    std::cout << "Demo serving on http://127.0.0.1:" << srv.port() << "/\n";
    // SIGINT/SIGTERM → 置位停止标志（仅做 async-signal-safe 的 sig_atomic_t 赋值）。
    // stop 声明为 static：非捕获 lambda 可调用其地址用作信号处理器，仍属本作用域。
    static volatile std::sig_atomic_t stop = 0;
    std::signal(SIGINT, [](int) { stop = 1; });
    std::signal(SIGTERM, [](int) { stop = 1; });
    while (!stop) std::this_thread::sleep_for(std::chrono::milliseconds(200));
    return 0;
}
