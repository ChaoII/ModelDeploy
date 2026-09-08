//
// demo_server —— 演示 Web 的后端可执行：按手写 manifest 提供目录元数据，并开 ServingServer。
//
// 懒加载语义：启动只读 manifest 目录（status=Unloaded、infer 空），不实例化任何模型；
// 单槽实例化经 ModelRepo::load 触达（前端切模型时懒加载）。HandleBuilder 按 manifest 的
// type 构造各族真实 SDK 模型（ORT CPU）。构造成功 → make_model_handle 产出非空 infer
// （status 由 ModelRepo::load 置 Ready）；构造失败/未初始化 → 返回空 infer 的元数据句柄，
// 由 load 统一置 Failed，绝不因单个模型崩溃进程。
#include <chrono>
#include <csignal>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <string>
#include <thread>

#include "serving/config.h"
#include "serving/model_repo.h"
#include "serving/model_entry.h"
#include "serving/manifest.h"
#include "serving/server.h"
#include "serving/adapters.h"

#include "runtime/runtime_option.h"
#include "vision/common/result_json.h"
#include "vision/detection/ultralytics_det.h"
#include "vision/classification/classification.h"
#include "vision/iseg/ultralytics_seg.h"
#include "vision/pose/ultralytics_pose.h"
#include "vision/obb/ultralytics_obb.h"
#include "vision/sem/ultralytics_sem.h"
#include "vision/depth/ultralytics_depth.h"
#include "vision/ocr/ppocr.h"
#include "vision/face/face_det/scrfd.h"
#include "vision/lpr/lpr_pipeline/lpr_pipeline.h"

using namespace modeldeploy;
using namespace modeldeploy::serving;

// 从 manifest 条目回填目录元数据（type/input_size/labels 已在扫描阶段解析）。
static ModelHandle meta_handle(const ManifestModel& m) {
    ModelHandle h;
    h.name = m.id;
    h.display = m.display;
    h.version = "1";
    h.type = m.type;
    h.input_size = m.input_size;
    h.labels = m.labels;
    return h;
}

// 构造成功 → make_model_handle 的核心句柄已带 infer，仅回填元数据后返回。
template <typename M>
static ModelHandle built_handle(const ManifestModel& m, std::unique_ptr<M> model) {
    ModelHandle h = make_model_handle<M>(m.id, std::move(model));
    h.display = m.display;
    h.type = m.type;
    h.input_size = m.input_size;
    h.labels = m.labels;
    return h;
}

// 按 manifest 条目构造真实模型句柄。files.* 已在 load_manifest 时拼上 base（如
// m.model_f/m.rec_f/m.cls_f/m.dict_f），故此处不再需要 base 参数（保留签名以对齐 API）。
int main(int argc, char** argv) {
    int port = 8000;
    std::string web_root, repo, base_arg;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto val = [&]() -> std::string { return (i + 1 < argc) ? argv[++i] : ""; };
        if (a == "--port") port = std::stoi(val());
        else if (a == "--web") web_root = val();
        else if (a == "--repo") repo = val();
        else if (a == "--base") base_arg = val();  // 兼容 CLI；实际根由 manifest 的 base 字段决定
        else if (a == "--help") {
            std::cout << "usage: demo_server [--port <port>] [--web <dir>] [--repo <manifest.json>] "
                         "[--base <onnx-root>]\n";
            std::cout << "  --repo  manifest.json 路径（缺省 repository/application 下的 demo_manifest.json）\n";
            std::cout << "  --web   静态 web 资产目录（缺省 web_demo）\n";
            std::cout << "  --base  onnx 根（缺省由 manifest 的 base 字段决定；资源根以 manifest 为准）\n";
            return 0;
        }
    }
    if (web_root.empty()) web_root = "web_demo";  // 构建目录下的 web 资产
    if (repo.empty()) repo = "application/demo_manifest.json";
    (void)base_arg;

    ServingConfig cfg;
    cfg.host = "0.0.0.0";
    cfg.port = port;
    cfg.web_root = web_root;
    cfg.model_repo = repo;

    HandleBuilder builder = [](const ManifestModel& m, const std::string&) {
        ModelHandle meta = meta_handle(m);
        RuntimeOption opt;
        opt.use_ort_backend();
        opt.set_device(Device::CPU, 0);  // CPU 起步；有 GPU/权重时改 Device::GPU

        if (m.type == "det") {
            try {
                using MM = ResultModel<vision::detection::UltralyticsDet,
                                       std::vector<vision::DetectionResult>>;
                auto model = std::make_unique<MM>(m.model_f, opt);
                if (!model->is_initialized()) return meta;
                return built_handle(m, std::move(model));
            } catch (...) { return meta; }
        }
        if (m.type == "cls") {
            try {
                using MM = ClassifyAdapter<vision::ClassifyResult>;
                auto model = std::make_unique<MM>(m.model_f, opt);
                if (!model->is_initialized()) return meta;
                return built_handle(m, std::move(model));
            } catch (...) { return meta; }
        }
        if (m.type == "seg") {
            try {
                using MM = ResultModel<vision::detection::UltralyticsSeg,
                                       std::vector<vision::InstanceSegResult>>;
                auto model = std::make_unique<MM>(m.model_f, opt);
                if (!model->is_initialized()) return meta;
                return built_handle(m, std::move(model));
            } catch (...) { return meta; }
        }
        if (m.type == "pose") {
            try {
                using MM = ResultModel<vision::detection::UltralyticsPose,
                                       std::vector<vision::KeyPointsResult>>;
                auto model = std::make_unique<MM>(m.model_f, opt);
                if (!model->is_initialized()) return meta;
                return built_handle(m, std::move(model));
            } catch (...) { return meta; }
        }
        if (m.type == "obb") {
            try {
                using MM = ResultModel<vision::detection::UltralyticsObb,
                                       std::vector<vision::ObbResult>>;
                auto model = std::make_unique<MM>(m.model_f, opt);
                if (!model->is_initialized()) return meta;
                return built_handle(m, std::move(model));
            } catch (...) { return meta; }
        }
        if (m.type == "sem") {
            try {
                using MM = ResultModel<vision::detection::UltralyticsSem, vision::SemSegResult>;
                auto model = std::make_unique<MM>(m.model_f, opt);
                if (!model->is_initialized()) return meta;
                return built_handle(m, std::move(model));
            } catch (...) { return meta; }
        }
        if (m.type == "depth") {
            try {
                using MM = ResultModel<vision::detection::UltralyticsDepth, vision::DepthResult>;
                auto model = std::make_unique<MM>(m.model_f, opt);
                if (!model->is_initialized()) return meta;
                return built_handle(m, std::move(model));
            } catch (...) { return meta; }
        }
        if (m.type == "face") {
            try {
                using MM = ResultModel<vision::face::Scrfd,
                                       std::vector<vision::KeyPointsResult>>;
                auto model = std::make_unique<MM>(m.model_f, opt);
                if (!model->is_initialized()) return meta;
                return built_handle(m, std::move(model));
            } catch (...) { return meta; }
        }
        if (m.type == "ocr") {
            try {
                // PaddleOCR 构造收 (det, cls, rec, dict, option)。
                using MM = ResultModel<vision::ocr::PaddleOCR, vision::OCRResult>;
                if (m.rec_f.empty() || m.dict_f.empty()) return meta;
                auto model = std::make_unique<MM>(m.model_f, m.cls_f, m.rec_f, m.dict_f, opt);
                if (!model->is_initialized()) return meta;
                return built_handle(m, std::move(model));
            } catch (...) { return meta; }
        }
        if (m.type == "lpr") {
            try {
                // LprPipeline 构造收 (det, rec, option)。
                using MM = LprAdapter<std::vector<vision::LprResult>>;
                if (m.rec_f.empty()) return meta;
                auto model = std::make_unique<MM>(m.model_f, m.rec_f, opt);
                if (!model->is_initialized()) return meta;
                return built_handle(m, std::move(model));
            } catch (...) { return meta; }
        }
        return meta;  // 未知 type：目录内可列，load 置 Failed
    };

    std::string err;
    ServingServer srv(cfg, std::move(builder), &err);
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
