//
// 全模型 + 全后端 benchmark：det/cls/obb/pose/seg/sem/depth/face*/lpr/ocr/insightface
// + 各 pipeline，分解 pre/infer/post 耗时。后端由模型文件后缀自动推断。
// 运行: benchmark [模型标签]；默认跑全部可用模型。
//
#include <catch2/catch_test_macros.hpp>
#include <opencv2/core/mat.hpp>
#include <string>
#include <memory>
#include <vector>
#include <array>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <cmath>
#include <chrono>

#include "csrc/vision.h"
#include "csrc/utils/benchmark.h"
#include "csrc/vision/face/insightface/face_analysis.h"
#include "core/md_log.h"

namespace fs = std::filesystem;
using namespace modeldeploy;
using namespace modeldeploy::vision;

namespace {

    fs::path bench_data_dir() {
        const char* env = std::getenv("TEST_DATA_DIR");
        return (env && *env) ? fs::path(env) / "test_data" : fs::current_path() / "test_data";
    }

    bool has_file(const fs::path& p) { return fs::exists(p); }

    // 统一输出：tag + pre/infer/post/total 平均耗时
    void report(const std::string& tag, const std::vector<TimerArray>& runs) {
        if (runs.empty()) return;
        double pre = 0, infer = 0, post = 0, total = 0;
        for (const auto& t : runs) {
            pre += t.pre_timer.sum_ms();
            infer += t.infer_timer.sum_ms();
            post += t.post_timer.sum_ms();
            total += t.pre_timer.sum_ms() + t.infer_timer.sum_ms() + t.post_timer.sum_ms();
        }
        const size_t n = runs.size();
        std::cout << "[bench] " << tag << " | pre=" << pre / n
                  << "ms infer=" << infer / n << "ms post=" << post / n
                  << "ms total=" << total / n << "ms (n=" << n << ")" << std::endl;
    }

    ImageData load_img(const std::string& name) {
        auto p = bench_data_dir() / "test_images" / name;
        if (!has_file(p)) return ImageData();
        return ImageData::imread(p.string());
    }

    // 找 ocr dict
    fs::path ocr_dict() {
        for (const auto& d : {"ppocrv4_dict.txt", "ppocrv5_dict.txt", "dict.txt"}) {
            auto p = bench_data_dir() / d;
            if (has_file(p)) return p;
        }
        return {};
    }

    // 找 ocr 模型（det/cls/rec）
    fs::path find_ocr_model(const std::string& kind, const std::string& ext) {
        for (const auto& v : {"ppocrv4_mobile", "ppocrv5_mobile", "ppocrv6_tiny", "repsvtr_mobile"}) {
            auto p = bench_data_dir() / "test_models" / "onnx" / "ocr" / v / (kind + "_infer" + ext);
            if (has_file(p)) return p;
        }
        return {};
    }
} // namespace

// ==================== 单模型 ====================

    // 后端选择：
    //  - .engine → TRT backend（GPU，最快，生产推荐）
    //  - .mnn    → MNN（CPU）
    //  - .onnx   → ORT CPU（基线；GPU 生产见 TRT backend / ORT TRT-EP 单独标注）
    // 注：ORT TRT-EP（enable_trt=true）对动态 shape 的 onnx（非 NMS 导出）极慢/卡死，
    //     故 onnx 统一 ORT CPU 作基线，GPU 用 trtexec 预编译 .engine（TRT backend）。
    RuntimeOption bench_opt(const std::string& rel) {
        RuntimeOption opt;
        if (rel.find(".engine") != std::string::npos) {
            opt.use_gpu(0);
            opt.use_trt_backend();
        } else {
            opt.use_cpu();
        }
        return opt;
    }

    // 当前构建是否支持 TRT（CPU 构建无 TRT，遇 .engine 必须跳过，否则 FATAL）
    bool bench_supported(const std::string& rel) {
#ifdef ENABLE_TRT
        return true;
#else
        return rel.find(".engine") == std::string::npos;
#endif
    }

    // onnx 基线：ORT CPU（GPU 用 TRT backend .engine）
    RuntimeOption bench_opt_onnx() {
        RuntimeOption opt;
        opt.use_cpu();
        return opt;
    }

// ==================== 统一：同一模型（yolo26n 家族）跨后端 benchmark ====================
// 目标：同一模型在不同后端上测性能，横向对比后端间差距。sophgo 仅在算能设备构建上启用。
enum class OpBackend { OrtCpu, MnnCpu, OrtCuda, OrtTrtEp, TrtEngine, Sophgo };

struct BenchSpec { OpBackend op; const char* tag; };

// 各后端顺序即输出顺序。
// 注意：ORT-TRT-EP 在本地对 yolo26n 动态 onnx 的首次 engine 构建时会卡死/崩溃（见注释），
// 默认不启用，需显式设环境变量 MODELDEPLOY_BENCH_TRT_EP=1 才纳入。
static std::vector<BenchSpec> bench_backends() {
    std::vector<BenchSpec> v;
#ifdef ENABLE_ORT
    v.push_back({OpBackend::OrtCpu, "ORT-CPU"});
    v.push_back({OpBackend::OrtCuda, "ORT-CUDA-EP"});
    const char* env_ep = std::getenv("MODELDEPLOY_BENCH_TRT_EP");
    if (env_ep && std::string(env_ep) == "1") {
        v.push_back({OpBackend::OrtTrtEp, "ORT-TRT-EP"});
    }
#endif
#ifdef ENABLE_MNN
    v.push_back({OpBackend::MnnCpu, "MNN-CPU"});
#endif
#ifdef ENABLE_TRT
    v.push_back({OpBackend::TrtEngine, "TRT-engine"});
#endif
#ifdef ENABLE_SOPHGO
    v.push_back({OpBackend::Sophgo, "SOPHGO-TPU"});
#endif
    return v;
}

// 按后端返回模型文件相对路径（onnx/mnn/engine/bmodel）
static fs::path bench_rel(const char* family, const char* name, OpBackend op) {
    switch (op) {
        case OpBackend::OrtCpu:
        case OpBackend::OrtCuda:
        case OpBackend::OrtTrtEp:
            return fs::path("onnx") / family / (std::string(name) + ".onnx");
        case OpBackend::MnnCpu:
            return fs::path("mnn") / family / (std::string(name) + ".mnn");
        case OpBackend::TrtEngine:
            return fs::path("trt") / family / (std::string(name) + ".engine");
        case OpBackend::Sophgo:
            return fs::path("sophgo") / family / (std::string(name) + ".bmodel");
    }
    return {};
}

// 生成对应 RuntimeOption；input 为方形输入边长（TRT-EP 静态 shape 用）
static RuntimeOption bench_opt(const BenchSpec& s, int input) {
    RuntimeOption opt;
    switch (s.op) {
        case OpBackend::OrtCpu:
            opt.use_ort_backend(); opt.use_cpu(); break;
        case OpBackend::MnnCpu:
            opt.use_mnn_backend(); opt.use_cpu(); break;
        case OpBackend::OrtCuda:
            opt.use_ort_backend(); opt.use_gpu(0); break;
        case OpBackend::OrtTrtEp:
            opt.use_ort_backend(); opt.use_gpu(0); opt.enable_trt = true;
            {  // 静态 shape，避免动态 onnx 在 ORT-TRT-EP 上反复构建/卡死
                const std::string sh = "1x3x" + std::to_string(input) + "x" + std::to_string(input);
                opt.set_trt_min_shape(sh); opt.set_trt_opt_shape(sh); opt.set_trt_max_shape(sh);
            }
            break;
        case OpBackend::TrtEngine:
            opt.use_trt_backend(); opt.use_gpu(0); break;
        case OpBackend::Sophgo:
            opt.use_sophgo_backend(0); break;
    }
    return opt;
}

// 通用计时：先预热（含 TRT-EP/engine 一次性构建，不计时），再 kRuns 次计时，
// 经 report() 输出 pre/infer/post 分解（infer 即纯后端推理耗时，GPU 后端为 GPU 算时）。
// run 返回值表示该帧结果是否有效（false 则终止该后端计时）。
template <typename Model, typename Run>
void bench_yolo(const char* family, const char* name, int input, const char* imgname, Run run) {
    constexpr int kRuns = 20;
    for (const auto& s : bench_backends()) {
        auto rel = bench_rel(family, name, s.op);
        auto mp = bench_data_dir() / "test_models" / rel;
        if (!has_file(mp)) {
            std::printf("[bench][skip] %-12s %s (missing)\n", s.tag, rel.string().c_str());
            continue;
        }
        try {
            RuntimeOption opt = bench_opt(s, input);
            Model model(mp.string(), opt);
            if (!model.is_initialized()) {
                std::printf("[bench][skip] %-12s %s (init fail)\n", s.tag, rel.string().c_str());
                continue;
            }
            auto img = load_img(imgname);
            if (img.empty()) continue;
            { for (int i = 0; i < 5; ++i) run(model, img, nullptr); }  // 预热
            std::vector<TimerArray> runs;
            for (int i = 0; i < kRuns; ++i) {
                TimerArray t;
                if (!run(model, img, &t)) { runs.clear(); break; }
                runs.push_back(t);
            }
            if (!runs.empty()) report(std::string(name) + " " + s.tag, runs);
        } catch (const std::exception& e) {
            std::printf("[bench][error] %-12s %s (%s)\n", s.tag, rel.string().c_str(), e.what());
        } catch (...) {
            std::printf("[bench][error] %-12s %s (unknown exception)\n", s.tag, rel.string().c_str());
        }
    }
}

// 纯墙钟总量计时（predict 无 TimerArray 的模型用，如 Classification）
template <typename Model, typename Run>
void bench_yolo_total_only(const char* family, const char* name, int input, const char* imgname, Run run) {
    constexpr int kRuns = 20;
    for (const auto& s : bench_backends()) {
        auto rel = bench_rel(family, name, s.op);
        auto mp = bench_data_dir() / "test_models" / rel;
        if (!has_file(mp)) {
            std::printf("[bench][skip] %-12s %s (missing)\n", s.tag, rel.string().c_str());
            continue;
        }
        try {
            RuntimeOption opt = bench_opt(s, input);
            Model model(mp.string(), opt);
            if (!model.is_initialized()) {
                std::printf("[bench][skip] %-12s %s (init fail)\n", s.tag, rel.string().c_str());
                continue;
            }
            auto img = load_img(imgname);
            if (img.empty()) continue;
            for (int i = 0; i < 5; ++i) run(model, img);  // 预热
            std::vector<double> times;
            for (int i = 0; i < kRuns; ++i) {
                auto t0 = std::chrono::high_resolution_clock::now();
                const bool ok = run(model, img);
                auto t1 = std::chrono::high_resolution_clock::now();
                if (!ok) { times.clear(); break; }
                times.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
            }
            if (!times.empty()) {
                double sum = 0;
                for (auto v : times) sum += v;
                std::printf("[bench] %-14s %-12s | total=%.3fms (n=%zu)\n",
                            name, s.tag, sum / times.size(), times.size());
            }
        } catch (const std::exception& e) {
            std::printf("[bench][error] %-12s %s (%s)\n", s.tag, rel.string().c_str(), e.what());
        } catch (...) {
            std::printf("[bench][error] %-12s %s (unknown exception)\n", s.tag, rel.string().c_str());
        }
    }
}

TEST_CASE("Benchmark UltralyticsDet", "[all_models][benchmark]") {
    bench_yolo<detection::UltralyticsDet>("yolo26n", "yolo26n", 640, "test_detection0.jpg",
        [](detection::UltralyticsDet& m, const ImageData& img, TimerArray* t) {
            std::vector<DetectionResult> r;
            return m.predict(img, &r, t) && !r.empty();
        });
}

TEST_CASE("Benchmark UltralyticsCls", "[all_models][benchmark]") {
    bench_yolo_total_only<classification::Classification>("yolo26n", "yolo26n-cls", 224, "test_person.jpg",
        [](classification::Classification& m, const ImageData& img) {
            ClassifyResult r;
            return m.predict(img, &r) && !r.label_ids.empty();
        });
}

TEST_CASE("Benchmark UltralyticsObb", "[all_models][benchmark]") {
    bench_yolo<detection::UltralyticsObb>("yolo26n", "yolo26n-obb", 640, "test_obb.jpg",
        [](detection::UltralyticsObb& m, const ImageData& img, TimerArray* t) {
            std::vector<ObbResult> r;
            return m.predict(img, &r, t) && !r.empty();
        });
}

TEST_CASE("Benchmark UltralyticsPose", "[all_models][benchmark]") {
    bench_yolo<detection::UltralyticsPose>("yolo26n", "yolo26n-pose", 640, "test_person.jpg",
        [](detection::UltralyticsPose& m, const ImageData& img, TimerArray* t) {
            std::vector<KeyPointsResult> r;
            return m.predict(img, &r, t) && !r.empty();
        });
}

TEST_CASE("Benchmark UltralyticsSeg", "[all_models][benchmark]") {
    bench_yolo<detection::UltralyticsSeg>("yolo26n", "yolo26n-seg", 640, "test_person.jpg",
        [](detection::UltralyticsSeg& m, const ImageData& img, TimerArray* t) {
            std::vector<InstanceSegResult> r;
            return m.predict(img, &r, t) && !r.empty();
        });
}

TEST_CASE("Benchmark UltralyticsDepth", "[all_models][benchmark]") {
    bench_yolo<detection::UltralyticsDepth>("yolo26n", "yolo26n-depth", 640, "test_person.jpg",
        [](detection::UltralyticsDepth& m, const ImageData& img, TimerArray* t) {
            DepthResult r;
            return m.predict(img, &r, t) && !r.depth.empty();
        });
}

TEST_CASE("Benchmark UltralyticsSem", "[all_models][benchmark]") {
    bench_yolo<detection::UltralyticsSem>("yolo26n", "yolo26n-sem", 640, "test_person.jpg",
        [](detection::UltralyticsSem& m, const ImageData& img, TimerArray* t) {
            SemSegResult r;
            return m.predict(img, &r, t) && !r.labels.empty();
        });
}

// ==================== 人脸模型 ====================

TEST_CASE("Benchmark Scrfd face det", "[all_models][benchmark]") {
    // onnx → ORT（GPU 下 TRT EP）；.engine → TRT backend
    for (const auto& rel : {"onnx/face/scrfd_2.5g_bnkps_shape640x640.onnx", "trt/scrfd_2.5g.engine"}) {
        auto mp = bench_data_dir() / "test_models" / rel;
        if (!has_file(mp)) continue;
        if (!bench_supported(rel)) continue;
        RuntimeOption opt = bench_opt(rel);
        face::Scrfd model(mp.string(), opt);
        if (!model.is_initialized()) continue;
        auto img = load_img("test_face_detection.jpg");
        if (img.empty()) continue;
        // 预热
        { std::vector<KeyPointsResult> r; for (int i = 0; i < 3; ++i) model.predict(img, &r); }
        constexpr int kRuns = 20;
        std::vector<TimerArray> runs;
        for (int i = 0; i < kRuns; ++i) {
            std::vector<KeyPointsResult> r;
            TimerArray t;
            REQUIRE(model.predict(img, &r, &t));
            if (r.empty()) { runs.clear(); break; }
            runs.push_back(t);
        }
        report(std::string("face-det ") + rel, runs);
    }
}

TEST_CASE("Benchmark SeetaFace face models", "[all_models][benchmark]") {
    // tag / onnx 文件 / trt engine 文件 / 图片
    struct FaceCfg { const char* tag; const char* file; const char* trt_file; const char* img; };
    const FaceCfg cfgs[] = {
        {"face-age", "age_predictor.onnx", "", "test_face_gender.jpg"},            // age 无法转 TRT（Gemm 固定 batch）
        {"face-gender", "gender_predictor.onnx", "gender_predictor.engine", "test_face_gender.jpg"},
        {"face-rec", "face_recognizer.onnx", "face_recognizer.engine", "test_face_id.jpg"},
    };
    for (const auto& c : cfgs) {
        // 遍历后端：onnx + trt engine
        std::vector<std::string> rels;
        rels.push_back(std::string("onnx/face/") + c.file);
        if (std::string(c.trt_file).size()) rels.push_back(std::string("trt/") + c.trt_file);
        for (const auto& rel : rels) {
            auto mp = bench_data_dir() / "test_models" / rel;
            if (!has_file(mp)) continue;
            if (!bench_supported(rel)) continue;
            RuntimeOption opt = bench_opt(rel);
            auto img = load_img(c.img);
            if (img.empty()) continue;
            constexpr int kRuns = 20;
            std::vector<TimerArray> runs;
            if (std::string(c.tag) == "face-age") {
                face::SeetaFaceAge model(mp.string(), opt);
                if (!model.is_initialized()) continue;
                { int a; for (int i = 0; i < 3; ++i) model.predict(img, &a); }
                for (int i = 0; i < kRuns; ++i) {
                    int age = 0;
                    TimerArray t;
                    REQUIRE(model.predict(img, &age, &t));
                    runs.push_back(t);
                }
            } else if (std::string(c.tag) == "face-gender") {
                face::SeetaFaceGender model(mp.string(), opt);
                if (!model.is_initialized()) continue;
                { int g; for (int i = 0; i < 3; ++i) model.predict(img, &g); }
                for (int i = 0; i < kRuns; ++i) {
                    int g = 0;
                    TimerArray t;
                    REQUIRE(model.predict(img, &g, &t));
                    runs.push_back(t);
                }
            } else {
                face::SeetaFaceID model(mp.string(), opt);
                if (!model.is_initialized()) continue;
                { FaceRecognitionResult r; for (int i = 0; i < 3; ++i) model.predict(img, &r); }
                for (int i = 0; i < kRuns; ++i) {
                    FaceRecognitionResult r;
                    TimerArray t;
                    REQUIRE(model.predict(img, &r, &t));
                    runs.push_back(t);
                }
            }
            report(std::string("face ") + c.tag + " " + rel, runs);
        }
    }
}

// ==================== LPR / OCR ====================

TEST_CASE("Benchmark LPR", "[all_models][benchmark]") {
    // lpr det（onnx + TRT backend）
    for (const auto& rel : {"onnx/yolov5plate.onnx", "trt/yolov5plate.engine"}) {
        auto mp = bench_data_dir() / "test_models" / rel;
        if (!has_file(mp)) continue;
        if (!bench_supported(rel)) continue;
        RuntimeOption opt = bench_opt(rel);
        lpr::LprDetection model(mp.string(), opt);
        if (!model.is_initialized()) continue;
        auto img = load_img("test_lpr_detection.jpg");
        if (img.empty()) continue;
        { std::vector<KeyPointsResult> r; for (int i = 0; i < 3; ++i) model.predict(img, &r); }
        constexpr int kRuns = 20;
        std::vector<TimerArray> runs;
        for (int i = 0; i < kRuns; ++i) {
            std::vector<KeyPointsResult> r;
            TimerArray t;
            REQUIRE(model.predict(img, &r, &t));
            if (r.empty()) { runs.clear(); break; }
            runs.push_back(t);
        }
        report(std::string("lpr-det ") + rel, runs);
    }
    // lpr rec（onnx + TRT backend）
    for (const auto& rel : {"onnx/plate_recognition_color.onnx", "trt/plate_recognition_color.engine"}) {
        auto mp = bench_data_dir() / "test_models" / rel;
        if (!has_file(mp)) continue;
        if (!bench_supported(rel)) continue;
        RuntimeOption opt = bench_opt(rel);
        lpr::LprRecognizer model(mp.string(), opt);
        if (!model.is_initialized()) continue;
        auto img = load_img("test_lpr_recognizer.jpg");
        if (img.empty()) continue;
        { LprResult r; for (int i = 0; i < 3; ++i) model.predict(img, &r); }
        constexpr int kRuns = 20;
        std::vector<TimerArray> runs;
        for (int i = 0; i < kRuns; ++i) {
            LprResult r;
            TimerArray t;
            REQUIRE(model.predict(img, &r, &t));
            runs.push_back(t);
        }
        report(std::string("lpr-rec ") + rel, runs);
    }
}

TEST_CASE("Benchmark OCR single models", "[all_models][benchmark]") {
    auto img = load_img("test_ocr.png");
    auto dict = ocr_dict();
    if (img.empty() || dict.empty()) return;

    // det（onnx + TRT backend）
    for (const auto& rel : {std::string("onnx/ocr/ppocrv4_mobile/det_infer.onnx"), std::string("trt/ocr_det.engine")}) {
        auto mp = bench_data_dir() / "test_models" / rel;
        if (!has_file(mp)) continue;
        if (!bench_supported(mp.string())) continue;
        RuntimeOption opt = bench_opt(mp.string());
        ocr::DBDetector model(mp.string(), opt);
        if (!model.is_initialized()) continue;
        { std::vector<std::array<int, 8>> b; for (int i = 0; i < 3; ++i) model.predict(img, &b); }
        constexpr int kRuns = 20;
        std::vector<TimerArray> runs;
        for (int i = 0; i < kRuns; ++i) {
            std::vector<std::array<int, 8>> boxes;
            TimerArray t;
            REQUIRE(model.predict(img, &boxes, &t));
            if (boxes.empty()) { runs.clear(); break; }
            runs.push_back(t);
        }
        report(std::string("ocr-det ") + rel, runs);
    }
    // rec（onnx + TRT backend）
    for (const auto& rel : {std::string("onnx/ocr/ppocrv4_mobile/rec_infer.onnx"), std::string("trt/ocr_rec.engine")}) {
        auto mp = bench_data_dir() / "test_models" / rel;
        if (!has_file(mp)) continue;
        if (!bench_supported(mp.string())) continue;
        RuntimeOption opt = bench_opt(mp.string());
        ocr::Recognizer model(mp.string(), dict.string(), opt);
        if (!model.is_initialized()) continue;
        { std::string t; float s; for (int i = 0; i < 3; ++i) model.predict(img, &t, &s); }
        constexpr int kRuns = 20;
        std::vector<TimerArray> runs;
        for (int i = 0; i < kRuns; ++i) {
            std::string text; float score = 0;
            TimerArray t;
            REQUIRE(model.predict(img, &text, &score, &t));
            if (text.empty()) { runs.clear(); break; }
            runs.push_back(t);
        }
        report(std::string("ocr-rec ") + rel, runs);
    }
    // cls（onnx + TRT backend）
    for (const auto& rel : {std::string("onnx/ocr/ppocrv4_mobile/cls_infer.onnx"), std::string("trt/ocr_cls.engine")}) {
        auto mp = bench_data_dir() / "test_models" / rel;
        if (!has_file(mp)) continue;
        if (!bench_supported(mp.string())) continue;
        RuntimeOption opt = bench_opt(mp.string());
        ocr::Classifier model(mp.string(), opt);
        if (!model.is_initialized()) continue;
        { int32_t l; float s; for (int i = 0; i < 3; ++i) model.predict(img, &l, &s); }
        constexpr int kRuns = 20;
        std::vector<double> times;
        for (int i = 0; i < kRuns; ++i) {
            int32_t label = -1; float score = 0;
            auto t0 = std::chrono::high_resolution_clock::now();
            REQUIRE(model.predict(img, &label, &score));
            auto t1 = std::chrono::high_resolution_clock::now();
            times.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
        }
        if (!times.empty()) {
            double sum = 0;
            for (auto v : times) sum += v;
            std::cout << "[bench] ocr-cls " << rel << " | total=" << sum / times.size()
                      << "ms (n=" << times.size() << ")" << std::endl;
        }
    }
}

// ==================== insightface 单模型 ====================

TEST_CASE("Benchmark insightface single models", "[all_models][benchmark]") {
    const std::string subdir = "test_models";
    struct IF { const char* tag; const char* file; int size; const char* img; };
    const IF ifs[] = {
        {"det", "det_10g.onnx", 640, "test_person.jpg"},
        {"lmk2d", "2d106det.onnx", 192, "test_person.jpg"},
        {"lmk3d", "1k3d68.onnx", 192, "test_person.jpg"},
        {"rec", "w600k_r50.onnx", 112, "test_person.jpg"},
        {"genderage", "genderage.onnx", 96, "test_person.jpg"},
    };
    auto img = load_img("test_person.jpg");
    if (img.empty()) return;
    // 用 ref 里第一张脸的 bbox
    std::array<float, 4> bbox{640.0f, 124.0f, 672.0f, 168.0f};

    for (const auto& c : ifs) {
        for (const auto& be : {std::string("onnx"), std::string("mnn"), std::string("trt")}) {
            // basename + 按后端扩展名
            std::string base(c.file);
            auto dot = base.find(".onnx");
            if (dot != std::string::npos) base = base.substr(0, dot);
            std::string ext = (be == "onnx") ? ".onnx" : (be == "mnn" ? ".mnn" : ".engine");
            auto mp = bench_data_dir() / subdir / be / "insightface" / "buffalo_l" / (base + ext);
            if (!has_file(mp)) continue;
            if (!bench_supported(mp.string())) continue;
            RuntimeOption opt = bench_opt(mp.string());
            std::vector<TimerArray> runs;
            constexpr int kRuns = 10;
            {   // 预热
                if (std::string(c.tag) == "det") {
                    face::InsightFaceDet m(mp.string(), opt);
                    if (!m.is_initialized()) continue;
                    std::vector<face::InsightFaceBox> r; for (int i = 0; i < 3; ++i) m.predict(img, &r);
                } else if (std::string(c.tag) == "lmk2d" || std::string(c.tag) == "lmk3d") {
                    face::InsightFaceLandmark m(mp.string(), opt);
                    if (!m.is_initialized()) continue;
                    std::vector<std::array<float, 2>> pts; for (int i = 0; i < 3; ++i) m.predict_2d106(img, bbox, &pts);
                } else if (std::string(c.tag) == "rec") {
                    face::InsightFaceRecognition m(mp.string(), opt);
                    if (!m.is_initialized()) continue;
                    std::vector<std::array<float, 2>> kps(5, {650.0f, 140.0f});
                    std::vector<float> emb; for (int i = 0; i < 3; ++i) m.predict(img, kps, &emb);
                } else {
                    face::InsightFaceGenderAge m(mp.string(), opt);
                    if (!m.is_initialized()) continue;
                    face::GenderAgeResult r; for (int i = 0; i < 3; ++i) m.predict_gender_age(img, bbox, &r);
                }
            }
            if (std::string(c.tag) == "det") {
                face::InsightFaceDet model(mp.string(), opt);
                if (!model.is_initialized()) continue;
                for (int i = 0; i < kRuns; ++i) {
                    std::vector<face::InsightFaceBox> r;
                    TimerArray t;
                    REQUIRE(model.predict(img, &r, &t));
                    if (r.empty()) { runs.clear(); break; }
                    runs.push_back(t);
                }
            } else if (std::string(c.tag) == "lmk2d") {
                face::InsightFaceLandmark model(mp.string(), opt);
                if (!model.is_initialized()) continue;
                for (int i = 0; i < kRuns; ++i) {
                    std::vector<std::array<float, 2>> pts;
                    TimerArray t;
                    REQUIRE(model.predict_2d106(img, bbox, &pts, &t));
                    runs.push_back(t);
                }
            } else if (std::string(c.tag) == "lmk3d") {
                face::InsightFaceLandmark model(mp.string(), opt);
                if (!model.is_initialized()) continue;
                for (int i = 0; i < kRuns; ++i) {
                    std::vector<std::array<float, 3>> pts;
                    std::array<float, 3> pose{0, 0, 0};
                    TimerArray t;
                    REQUIRE(model.predict_3d68(img, bbox, &pts, &pose, &t));
                    runs.push_back(t);
                }
            } else if (std::string(c.tag) == "rec") {
                face::InsightFaceRecognition model(mp.string(), opt);
                if (!model.is_initialized()) continue;
                std::vector<std::array<float, 2>> kps(5, {650.0f, 140.0f});
                for (int i = 0; i < kRuns; ++i) {
                    std::vector<float> emb;
                    TimerArray t;
                    REQUIRE(model.predict(img, kps, &emb, &t));
                    runs.push_back(t);
                }
            } else {
                face::InsightFaceGenderAge model(mp.string(), opt);
                if (!model.is_initialized()) continue;
                for (int i = 0; i < kRuns; ++i) {
                    face::GenderAgeResult r;
                    TimerArray t;
                    REQUIRE(model.predict_gender_age(img, bbox, &r, &t));
                    runs.push_back(t);
                }
            }
            report(std::string("insightface ") + c.tag + " " + be, runs);
        }
    }
}

// ==================== pipeline ====================

TEST_CASE("Benchmark LPR pipeline", "[pipeline][benchmark]") {
    // onnx（ORT CPU 基线）+ trt（TRT backend）
    for (const auto& be : {std::string("onnx"), std::string("trt")}) {
        auto det = (be == "trt") ? (bench_data_dir() / "test_models" / "trt" / "yolov5plate.engine")
                                 : (bench_data_dir() / "test_models" / "onnx" / "yolov5plate.onnx");
        auto rec = (be == "trt") ? (bench_data_dir() / "test_models" / "trt" / "plate_recognition_color.engine")
                                 : (bench_data_dir() / "test_models" / "onnx" / "plate_recognition_color.onnx");
        if (!has_file(det) || !has_file(rec)) continue;
        RuntimeOption opt = bench_opt(det.string());
        lpr::LprPipeline model(det.string(), rec.string(), opt);
        if (!model.is_initialized()) continue;
        auto img = load_img("test_lpr_pipeline.jpg");
        if (img.empty()) continue;
        { std::vector<LprResult> r; for (int i = 0; i < 2; ++i) model.predict(img, &r); }
        constexpr int kRuns = 20;
        std::vector<TimerArray> runs;
        for (int i = 0; i < kRuns; ++i) {
            std::vector<LprResult> r;
            TimerArray t;
            REQUIRE(model.predict(img, &r, &t));
            runs.push_back(t);
        }
        report("pipeline lpr (det+rec) " + be, runs);
    }
}

TEST_CASE("Benchmark face recognition pipeline", "[pipeline][benchmark]") {
    // onnx（ORT CPU 基线）+ trt（TRT backend）
    for (const auto& be : {std::string("onnx"), std::string("trt")}) {
        auto sub = bench_data_dir() / "test_models" / be;
        auto det = (be == "trt") ? (sub / "scrfd_2.5g.engine")
                                 : (sub / "face" / "scrfd_2.5g_bnkps_shape640x640.onnx");
        auto rec = (be == "trt") ? (sub / "face_recognizer.engine")
                                 : (sub / "face" / "face_recognizer.onnx");
        if (!has_file(det) || !has_file(rec)) continue;
        RuntimeOption opt = bench_opt(det.string());
        face::FaceRecognizerPipeline model(det.string(), rec.string(), opt);
        if (!model.is_initialized()) continue;
        auto img = load_img("test_face_detection.jpg");
        if (img.empty()) continue;
        { std::vector<FaceRecognitionResult> r; for (int i = 0; i < 3; ++i) model.predict(img, &r); }
        constexpr int kRuns = 20;
        std::vector<TimerArray> runs;
        for (int i = 0; i < kRuns; ++i) {
            std::vector<FaceRecognitionResult> r;
            TimerArray t;
            REQUIRE(model.predict(img, &r, &t));
            if (r.empty()) { runs.clear(); break; }
            runs.push_back(t);
        }
        report("pipeline face-rec (scrfd+rec) " + be, runs);
    }
}

TEST_CASE("Benchmark face anti-spoof pipeline", "[pipeline][benchmark]") {
    auto det = bench_data_dir() / "test_models" / "onnx" / "face" / "scrfd_2.5g_bnkps_shape640x640.onnx";
    auto first = bench_data_dir() / "test_models" / "onnx" / "face" / "fas_first.onnx";
    auto second = bench_data_dir() / "test_models" / "onnx" / "face" / "fas_second.onnx";
    if (!has_file(det) || !has_file(first) || !has_file(second)) return;
    RuntimeOption opt = bench_opt_onnx();
    face::SeetaFaceAsPipeline model(det.string(), first.string(), second.string(), opt);
    if (!model.is_initialized()) return;
    auto img = load_img("test_face_detection.jpg");
    if (img.empty()) return;
    constexpr int kRuns = 20;
    std::vector<double> times;
    for (int i = 0; i < kRuns; ++i) {
        std::vector<FaceAntiSpoofResult> r;
        auto t0 = std::chrono::high_resolution_clock::now();
        REQUIRE(model.predict(img, &r));
        auto t1 = std::chrono::high_resolution_clock::now();
        times.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
    }
    if (!times.empty()) {
        double sum = 0;
        for (auto v : times) sum += v;
        std::cout << "[bench] pipeline face-as (scrfd+first+second) onnx | total=" << sum / times.size()
                  << "ms (n=" << times.size() << ")" << std::endl;
    }
}

TEST_CASE("Benchmark pedestrian attribute pipeline", "[pipeline][benchmark]") {
    // onnx（ORT CPU 基线）+ trt（TRT backend）
    for (const auto& be : {std::string("onnx"), std::string("trt")}) {
        auto det = (be == "trt") ? (bench_data_dir() / "test_models" / "trt" / "zhgd_det.engine")
                                 : (bench_data_dir() / "test_models" / "onnx" / "zhgd_det.onnx");
        auto ml = (be == "trt") ? (bench_data_dir() / "test_models" / "trt" / "zhgd_ml.engine")
                                : (bench_data_dir() / "test_models" / "onnx" / "zhgd_ml.onnx");
        if (!has_file(det) || !has_file(ml)) continue;
        RuntimeOption opt = bench_opt(det.string());
        pipeline::PedestrianAttribute model(det.string(), ml.string(), opt);
        if (!model.is_initialized()) continue;
        model.set_det_input_size({1280, 1280});
        model.set_cls_input_size({192, 256});
        auto img = load_img("test_pedestrian_attribute.jpg");
        if (img.empty()) continue;
        { std::vector<AttributeResult> r; for (int i = 0; i < 2; ++i) model.predict(img, &r); }
        constexpr int kRuns = 20;
        std::vector<TimerArray> runs;
        for (int i = 0; i < kRuns; ++i) {
            std::vector<AttributeResult> r;
            TimerArray t;
            REQUIRE(model.predict(img, &r, &t));
            if (r.empty()) { runs.clear(); break; }
            runs.push_back(t);
        }
        report("pipeline pedestrian-attr (det+mlc) " + be, runs);
    }
}

TEST_CASE("Benchmark OCR pipeline", "[pipeline][benchmark]") {
    auto dict = ocr_dict();
    if (dict.empty()) return;
    // onnx（ORT CPU 基线）+ trt（TRT backend）
    for (const auto& be : {std::string("onnx"), std::string("trt")}) {
        auto det = (be == "trt") ? (bench_data_dir() / "test_models" / "trt" / "ocr_det.engine")
                                 : find_ocr_model("det", ".onnx");
        auto cls = (be == "trt") ? (bench_data_dir() / "test_models" / "trt" / "ocr_cls.engine")
                                 : find_ocr_model("cls", ".onnx");
        auto rec = (be == "trt") ? (bench_data_dir() / "test_models" / "trt" / "ocr_rec.engine")
                                 : find_ocr_model("rec", ".onnx");
        if (!has_file(det) || !has_file(cls) || !has_file(rec)) continue;
        RuntimeOption opt = bench_opt(det.string());
        ocr::PaddleOCR model(det.string(), cls.string(), rec.string(), dict.string(), opt);
        if (!model.is_initialized()) continue;
        auto img = load_img("test_ocr.png");
        if (img.empty()) continue;
        { OCRResult r; for (int i = 0; i < 2; ++i) model.predict(img, &r); }
        constexpr int kRuns = 20;
        std::vector<TimerArray> runs;
        for (int i = 0; i < kRuns; ++i) {
            OCRResult r;
            TimerArray t;
            REQUIRE(model.predict(img, &r, &t));
            if (r.boxes.empty()) { runs.clear(); break; }
            runs.push_back(t);
        }
        report("pipeline ocr (det+cls+rec) " + be, runs);
    }
}

TEST_CASE("Benchmark insightface pipeline", "[pipeline][benchmark]") {
    for (const auto& be : {std::string("onnx"), std::string("mnn"), std::string("trt")}) {
        auto dir = bench_data_dir() / "test_models" / be / "insightface" / "buffalo_l";
        auto ext = (be == "onnx") ? ".onnx" : (be == "mnn" ? ".mnn" : ".engine");
        std::string d10 = std::string("det_10g") + ext;
        std::string w6 = std::string("w600k_r50") + ext;
        std::string lm2 = std::string("2d106det") + ext;
        std::string lm3 = std::string("1k3d68") + ext;
        std::string ga = std::string("genderage") + ext;
        auto det = dir / d10;
        if (!has_file(det)) continue;
        RuntimeOption opt;
        if (be == "trt") { opt.use_gpu(0); opt.use_trt_backend(); }
        else if (be == "onnx") { opt = bench_opt_onnx(); }
        else opt.use_cpu();
        auto analysis = std::make_unique<face::InsightFaceAnalysis>(
            (dir / d10).string(), (dir / w6).string(),
            (dir / lm2).string(), (dir / lm3).string(),
            opt, (dir / ga).string());
        if (!analysis->is_initialized()) continue;
        auto img = load_img("test_person.jpg");
        if (img.empty()) continue;
        constexpr int kRuns = 10;
        std::vector<TimerArray> runs;
        for (int i = 0; i < kRuns; ++i) {
            std::vector<face::InsightFaceResult> r;
            TimerArray t;
            REQUIRE(analysis->analyze(img, &r, true, true, true, true, &t));
            if (r.empty()) { runs.clear(); break; }
            runs.push_back(t);
        }
        report(std::string("pipeline insightface (det+lmk+rec+ga) ") + be, runs);
    }
}

#ifdef ENABLE_SOPHGO
// ==================== SOPHGO 后端（Linux + Sophon-Sail，BM1688/CV186AH） ====================
// yolo26n 家族单模型在 sophgo 后端上的 benchmark 已由上方统一的 bench_yolo() 覆盖
//（见 bench_backends() 中 SOPHGO-TPU 分支，bmodel 经 ENABLE_SOPHGO 时才启用）。

// SOPHGO insightface pipeline（det + 子模型，fp16/int8）
TEST_CASE("Benchmark SOPHGO insightface pipeline", "[sophgo][benchmark]") {
    auto dir = bench_data_dir() / "test_models" / "sophgo" / "insightface" / "buffalo_l";
    if (!has_file(dir / "det_10g_f16.bmodel")) return;
    RuntimeOption opt;
    opt.use_sophgo_backend(0);
    auto analysis = std::make_unique<face::InsightFaceAnalysis>(
        (dir / "det_10g_f16.bmodel").string(), (dir / "w600k_r50_f16.bmodel").string(),
        (dir / "2d106det_f16.bmodel").string(), (dir / "1k3d68_f16.bmodel").string(),
        opt, (dir / "genderage_f16.bmodel").string());
    if (!analysis->is_initialized()) return;
    auto img = load_img("test_person.jpg");
    if (img.empty()) return;
    constexpr int kRuns = 10;
    std::vector<TimerArray> runs;
    for (int i = 0; i < kRuns; ++i) {
        std::vector<face::InsightFaceResult> r;
        TimerArray t;
        REQUIRE(analysis->analyze(img, &r, true, true, true, true, &t));
        if (r.empty()) { runs.clear(); break; }
        runs.push_back(t);
    }
    report("sophgo pipeline insightface (det+lmk+rec+ga) f16", runs);
}
#endif // ENABLE_SOPHGO

// with-NMS(end2end) 模型：输出 [1,300,7]，模型内已做 NMS，SDK 走 run_with_nms（post≈0）。
// end2end 输入为 1024x1024，预处理器默认 640 需显式覆盖。仅 ORT-CPU 验证。
TEST_CASE("Benchmark UltralyticsObb end2end (with-NMS)", "[all_models][benchmark]") {
    constexpr int kRuns = 20;
    const auto rel = bench_rel("yolo26n", "yolo26n-obb-end2end", OpBackend::OrtCpu);
    const auto mp = bench_data_dir() / "test_models" / rel;
    if (!has_file(mp)) {
        std::printf("[bench][skip] ORT-CPU %s (missing)\n", rel.string().c_str());
        return;
    }
    try {
        RuntimeOption opt = bench_opt(BenchSpec{OpBackend::OrtCpu, "ORT-CPU"}, 1024);
        detection::UltralyticsObb model(mp.string(), opt);
        if (!model.is_initialized()) {
            std::printf("[bench][skip] ORT-CPU %s (init fail)\n", rel.string().c_str());
            return;
        }
        model.get_preprocessor().set_size({1024, 1024});
        auto img = load_img("test_obb.jpg");
        if (img.empty()) return;

        std::vector<ObbResult> result;
        if (!model.predict(img, &result)) {
            std::printf("[bench][error] yolo26n-obb-end2end predict failed\n");
            return;
        }
        std::printf("[bench] yolo26n-obb-end2end boxes=%zu\n", result.size());

        for (int i = 0; i < 5; ++i) model.predict(img, &result);  // 预热
        std::vector<TimerArray> runs;
        for (int i = 0; i < kRuns; ++i) {
            TimerArray t;
            if (!model.predict(img, &result, &t)) { runs.clear(); break; }
            runs.push_back(t);
        }
        if (!runs.empty()) report("yolo26n-obb-end2end ORT-CPU", runs);
    } catch (const std::exception& e) {
        std::printf("[bench][error] yolo26n-obb-end2end (%s)\n", e.what());
    } catch (...) {
        std::printf("[bench][error] yolo26n-obb-end2end (unknown exception)\n");
    }
}
