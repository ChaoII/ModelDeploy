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
            pre += t.pre_timer.average_ms();
            infer += t.infer_timer.average_ms();
            post += t.post_timer.average_ms();
            total += t.total_ms();
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

TEST_CASE("Benchmark UltralyticsDet", "[all_models][benchmark]") {
    // det 特殊：GPU(ORT TRT EP) 用内嵌 NMS 版（生产配置，~5ms）；
    // CPU ORT 用非 NMS 版（SDK 静态 ORT 加载 NMS onnx 报 protobuf 失败）
#ifdef WITH_GPU
    for (const auto& rel : {"onnx/yolo11n/yolo11n_nms.onnx", "trt/yolo11n_nms.engine"}) {
#else
    for (const auto& rel : {"onnx/yolo11n/yolo11n.onnx", "mnn/yolo11n_nms.mnn"}) {
#endif
        auto mp = bench_data_dir() / "test_models" / rel;
        if (!has_file(mp)) continue;
        if (!bench_supported(rel)) continue;
        RuntimeOption opt = bench_opt(rel);
        detection::UltralyticsDet model(mp.string(), opt);
        if (!model.is_initialized()) continue;
        auto img = load_img("test_detection0.jpg");
        if (img.empty()) continue;
        // 预热：触发 TRT EP engine 构建 / 内存分配等一次性开销（不计时）
        {
            std::vector<DetectionResult> r;
            for (int i = 0; i < 5; ++i) model.predict(img, &r);
        }
        constexpr int kRuns = 20;
        std::vector<TimerArray> runs;
        for (int i = 0; i < kRuns; ++i) {
            std::vector<DetectionResult> r;
            TimerArray t;
            REQUIRE(model.predict(img, &r, &t));
            if (r.empty()) { runs.clear(); break; }
            runs.push_back(t);
        }
        report(std::string("det ") + rel, runs);
    }
}

TEST_CASE("Benchmark UltralyticsCls", "[all_models][benchmark]") {
    for (const auto& rel : {"onnx/yolo11n/yolo11n-cls.onnx", "mnn/yolo11n-cls.mnn", "trt/yolo11n-cls.engine"}) {
        auto mp = bench_data_dir() / "test_models" / rel;
        if (!has_file(mp)) continue;
        if (!bench_supported(rel)) continue;
        RuntimeOption opt = bench_opt(rel);
        classification::Classification model(mp.string(), opt);
        if (!model.is_initialized()) continue;
        auto img = load_img("test_person.jpg");
        if (img.empty()) continue;
        constexpr int kRuns = 20;
        std::vector<double> times;
        for (int i = 0; i < kRuns; ++i) {
            ClassifyResult r;
            auto t0 = std::chrono::high_resolution_clock::now();
            REQUIRE(model.predict(img, &r));
            auto t1 = std::chrono::high_resolution_clock::now();
            if (r.label_ids.empty()) { times.clear(); break; }
            times.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
        }
        if (!times.empty()) {
            double sum = 0;
            for (auto v : times) sum += v;
            std::cout << "[bench] cls " << rel << " | total=" << sum / times.size()
                      << "ms (n=" << times.size() << ")" << std::endl;
        }
    }
}

TEST_CASE("Benchmark UltralyticsObb", "[all_models][benchmark]") {
    for (const auto& rel : {"onnx/yolo11n/yolo11n-obb.onnx", "mnn/yolo11n-obb_nms.mnn", "trt/yolo11n-obb_nms.engine"}) {
        auto mp = bench_data_dir() / "test_models" / rel;
        if (!has_file(mp)) continue;
        if (!bench_supported(rel)) continue;
        RuntimeOption opt = bench_opt(rel);
        detection::UltralyticsObb model(mp.string(), opt);
        if (!model.is_initialized()) continue;
        auto img = load_img("test_obb.jpg");
        if (img.empty()) continue;
        constexpr int kRuns = 20;
        std::vector<TimerArray> runs;
        for (int i = 0; i < kRuns; ++i) {
            std::vector<ObbResult> r;
            TimerArray t;
            REQUIRE(model.predict(img, &r, &t));
            if (r.empty()) { runs.clear(); break; }
            runs.push_back(t);
        }
        report(std::string("obb ") + rel, runs);
    }
}

TEST_CASE("Benchmark UltralyticsPose", "[all_models][benchmark]") {
    for (const auto& rel : {"onnx/yolo11n/yolo11n-pose.onnx", "mnn/yolo11n-pose_nms.mnn", "trt/yolo11n-pose_nms.engine"}) {
        auto mp = bench_data_dir() / "test_models" / rel;
        if (!has_file(mp)) continue;
        if (!bench_supported(rel)) continue;
        RuntimeOption opt = bench_opt(rel);
        detection::UltralyticsPose model(mp.string(), opt);
        if (!model.is_initialized()) continue;
        auto img = load_img("test_person.jpg");
        if (img.empty()) continue;
        constexpr int kRuns = 20;
        std::vector<TimerArray> runs;
        for (int i = 0; i < kRuns; ++i) {
            std::vector<KeyPointsResult> r;
            TimerArray t;
            REQUIRE(model.predict(img, &r, &t));
            if (r.empty()) { runs.clear(); break; }
            runs.push_back(t);
        }
        report(std::string("pose ") + rel, runs);
    }
}

TEST_CASE("Benchmark UltralyticsSeg", "[all_models][benchmark]") {
    for (const auto& rel : {"onnx/yolo11n/yolo11n-seg.onnx", "mnn/yolo11n-seg_nms.mnn", "trt/yolo11n-seg_nms.engine"}) {
        auto mp = bench_data_dir() / "test_models" / rel;
        if (!has_file(mp)) continue;
        if (!bench_supported(rel)) continue;
        RuntimeOption opt = bench_opt(rel);
        detection::UltralyticsSeg model(mp.string(), opt);
        if (!model.is_initialized()) continue;
        auto img = load_img("test_person.jpg");
        if (img.empty()) continue;
        constexpr int kRuns = 20;
        std::vector<TimerArray> runs;
        for (int i = 0; i < kRuns; ++i) {
            std::vector<InstanceSegResult> r;
            TimerArray t;
            REQUIRE(model.predict(img, &r, &t));
            if (r.empty()) { runs.clear(); break; }
            runs.push_back(t);
        }
        report(std::string("seg ") + rel, runs);
    }
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
// 遍历 test_models/sophgo/ 下所有已转换的 bmodel（fp16/int8），用对应模型类推理计时。
// 模型转换见 tools/docker/sophgo/convert_all.sh。
TEST_CASE("Benchmark SOPHGO models", "[sophgo][benchmark]") {
    auto soph_dir = bench_data_dir() / "test_models" / "sophgo";
    if (!fs::exists(soph_dir)) return;
    RuntimeOption opt;
    opt.use_sophgo_backend(0);

    // name -> 模型类构造（用 lambda 统一 predict 到 TimerArray）
    // 640 版 post 候选少 4x（8400 vs 33600）
    struct SG { const char* bmodel; const char* img; };
    const SG cfgs[] = {
        {"yolo11n.bmodel", "test_detection0.jpg"},                  // det 640 (84类)
        {"yolo11n_det1280_f16.bmodel", "test_detection0.jpg"},      // det 1280 f16 (单类5)
        {"yolo11n_det1280_int8.bmodel", "test_detection0.jpg"},     // det 1280 int8 (单类5)
        {"zhgd_without_nms_640.bmodel", "test_pedestrian_attribute.jpg"},   // zhgd 单类5
        {"zhgd_without_nms_1280.bmodel", "test_pedestrian_attribute.jpg"},  // zhgd 单类5 1280
        {"yolo11n-cls_f16.bmodel", "test_person.jpg"},
        {"yolo11n-cls_int8.bmodel", "test_person.jpg"},
        {"yolo11n-obb.bmodel", "test_obb.jpg"},                     // obb 640
        {"yolo11n-obb_f16.bmodel", "test_obb.jpg"},                 // obb 1024 f16
        {"yolo11n-obb_int8.bmodel", "test_obb.jpg"},                // obb 1024 int8
        {"yolo11n-pose.bmodel", "test_person.jpg"},                 // pose 640
        {"yolo11n-pose_f16.bmodel", "test_person.jpg"},
        {"yolo11n-pose_int8.bmodel", "test_person.jpg"},
        {"yolo11n-seg.bmodel", "test_person.jpg"},                  // seg 640
        {"yolo11n-seg_f16.bmodel", "test_person.jpg"},
        {"yolo11n-seg_int8.bmodel", "test_person.jpg"},
    };
    for (const auto& c : cfgs) {
        auto mp = soph_dir / c.bmodel;
        if (!has_file(mp)) continue;
        auto img = load_img(c.img);
        if (img.empty()) continue;
        std::vector<TimerArray> runs;
        constexpr int kRuns = 20;
        if (std::string(c.bmodel).find("-cls") != std::string::npos) {
            classification::Classification m(mp.string(), opt);
            if (!m.is_initialized()) continue;
            for (int i = 0; i < kRuns; ++i) {
                ClassifyResult r;
                TimerArray t;
                auto t0 = std::chrono::high_resolution_clock::now();
                REQUIRE(m.predict(img, &r));
                auto t1 = std::chrono::high_resolution_clock::now();
                TimerArray tt; tt.pre_timer.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
                runs.push_back(tt);
            }
        } else if (std::string(c.bmodel).find("-obb") != std::string::npos) {
            detection::UltralyticsObb m(mp.string(), opt);
            if (!m.is_initialized()) continue;
            // obb bmodel：yolo11n-obb.bmodel=640，_f16/_int8=1024
            const int obb_size = (std::string(c.bmodel).find("-obb_f16") != std::string::npos ||
                                  std::string(c.bmodel).find("-obb_int8") != std::string::npos) ? 1024 : 640;
            m.get_preprocessor().set_size({obb_size, obb_size});
            for (int i = 0; i < kRuns; ++i) {
                std::vector<ObbResult> r; TimerArray t;
                REQUIRE(m.predict(img, &r, &t));
                if (r.empty()) { runs.clear(); break; }
                runs.push_back(t);
            }
        } else if (std::string(c.bmodel).find("-pose") != std::string::npos) {
            detection::UltralyticsPose m(mp.string(), opt);
            if (!m.is_initialized()) continue;
            m.get_preprocessor().set_size({640, 640});
            for (int i = 0; i < kRuns; ++i) {
                std::vector<KeyPointsResult> r; TimerArray t;
                REQUIRE(m.predict(img, &r, &t));
                if (r.empty()) { runs.clear(); break; }
                runs.push_back(t);
            }
        } else if (std::string(c.bmodel).find("-seg") != std::string::npos) {
            detection::UltralyticsSeg m(mp.string(), opt);
            if (!m.is_initialized()) continue;
            m.get_preprocessor().set_size({640, 640});
            for (int i = 0; i < kRuns; ++i) {
                std::vector<InstanceSegResult> r; TimerArray t;
                REQUIRE(m.predict(img, &r, &t));
                if (r.empty()) { runs.clear(); break; }
                runs.push_back(t);
            }
        } else {
            detection::UltralyticsDet m(mp.string(), opt);
            if (!m.is_initialized()) continue;
            // det bmodel 输入：1280（yolo11n_det1280_*）或 640（yolo11n.bmodel）
            const int det_size = (std::string(c.bmodel).find("1280") != std::string::npos) ? 1280 : 640;
            m.get_preprocessor().set_size({det_size, det_size});
            for (int i = 0; i < kRuns; ++i) {
                std::vector<DetectionResult> r; TimerArray t;
                REQUIRE(m.predict(img, &r, &t));
                if (r.empty()) { runs.clear(); break; }
                runs.push_back(t);
            }
        }
        report(std::string("sophgo ") + c.bmodel, runs);
    }
}

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
