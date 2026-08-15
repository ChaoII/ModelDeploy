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

    // 后端选择：.engine → GPU TRT，否则 CPU（ORT/MNN 由扩展名自动）
    RuntimeOption bench_opt(const std::string& rel) {
        RuntimeOption opt;
        if (rel.find(".engine") != std::string::npos) { opt.use_gpu(0); opt.use_trt_backend(); }
        else opt.use_cpu();
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

TEST_CASE("Benchmark UltralyticsDet", "[all_models][benchmark]") {
    for (const auto& rel : {"onnx/yolo11n/yolo11n.onnx", "mnn/yolo11n_nms.mnn", "trt/yolo11n_nms.engine"}) {
        auto mp = bench_data_dir() / "test_models" / rel;
        if (!has_file(mp)) continue;
        if (!bench_supported(rel)) continue;
        RuntimeOption opt = bench_opt(rel);
        detection::UltralyticsDet model(mp.string(), opt);
        if (!model.is_initialized()) continue;
        auto img = load_img("test_detection0.jpg");
        if (img.empty()) continue;
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
    auto mp = bench_data_dir() / "test_models" / "onnx" / "face" / "scrfd_2.5g_bnkps_shape640x640.onnx";
    if (!has_file(mp)) return;
    RuntimeOption opt;
    opt.use_cpu();
    face::Scrfd model(mp.string(), opt);
    if (!model.is_initialized()) return;
    auto img = load_img("test_face_detection.jpg");
    if (img.empty()) return;
    constexpr int kRuns = 20;
    std::vector<TimerArray> runs;
    for (int i = 0; i < kRuns; ++i) {
        std::vector<KeyPointsResult> r;
        TimerArray t;
        REQUIRE(model.predict(img, &r, &t));
        if (r.empty()) { runs.clear(); break; }
        runs.push_back(t);
    }
    report("face-det scrfd onnx", runs);
}

TEST_CASE("Benchmark SeetaFace face models", "[all_models][benchmark]") {
    struct FaceCfg { const char* tag; const char* file; const char* img; };
    const FaceCfg cfgs[] = {
        {"face-age", "age_predictor.onnx", "test_face_gender.jpg"},
        {"face-gender", "gender_predictor.onnx", "test_face_gender.jpg"},
        {"face-rec", "face_recognizer.onnx", "test_face_id.jpg"},
    };
    for (const auto& c : cfgs) {
        auto mp = bench_data_dir() / "test_models" / "onnx" / "face" / c.file;
        if (!has_file(mp)) continue;
        RuntimeOption opt;
        opt.use_cpu();
        std::vector<TimerArray> runs;
        auto img = load_img(c.img);
        if (img.empty()) continue;
        constexpr int kRuns = 20;
        if (std::string(c.tag) == "face-age") {
            face::SeetaFaceAge model(mp.string(), opt);
            if (!model.is_initialized()) continue;
            for (int i = 0; i < kRuns; ++i) {
                int age = 0;
                TimerArray t;
                REQUIRE(model.predict(img, &age, &t));
                runs.push_back(t);
            }
        } else if (std::string(c.tag) == "face-gender") {
            face::SeetaFaceGender model(mp.string(), opt);
            if (!model.is_initialized()) continue;
            for (int i = 0; i < kRuns; ++i) {
                int g = 0;
                TimerArray t;
                REQUIRE(model.predict(img, &g, &t));
                runs.push_back(t);
            }
        } else {
            face::SeetaFaceID model(mp.string(), opt);
            if (!model.is_initialized()) continue;
            for (int i = 0; i < kRuns; ++i) {
                FaceRecognitionResult r;
                TimerArray t;
                REQUIRE(model.predict(img, &r, &t));
                runs.push_back(t);
            }
        }
        report(std::string("face ") + c.tag + " onnx", runs);
    }
}

// ==================== LPR / OCR ====================

TEST_CASE("Benchmark LPR", "[all_models][benchmark]") {
    // lpr det
    {
        auto mp = bench_data_dir() / "test_models" / "onnx" / "yolov5plate.onnx";
        if (has_file(mp)) {
            RuntimeOption opt; opt.use_cpu();
            lpr::LprDetection model(mp.string(), opt);
            if (model.is_initialized()) {
                auto img = load_img("test_lpr_detection.jpg");
                if (!img.empty()) {
                    constexpr int kRuns = 20;
                    std::vector<TimerArray> runs;
                    for (int i = 0; i < kRuns; ++i) {
                        std::vector<KeyPointsResult> r;
                        TimerArray t;
                        REQUIRE(model.predict(img, &r, &t));
                        if (r.empty()) { runs.clear(); break; }
                        runs.push_back(t);
                    }
                    report("lpr-det yolov5plate onnx", runs);
                }
            }
        }
    }
    // lpr rec
    {
        auto mp = bench_data_dir() / "test_models" / "onnx" / "plate_recognition_color.onnx";
        if (has_file(mp)) {
            RuntimeOption opt; opt.use_cpu();
            lpr::LprRecognizer model(mp.string(), opt);
            if (model.is_initialized()) {
                auto img = load_img("test_lpr_recognizer.jpg");
                if (!img.empty()) {
                    constexpr int kRuns = 20;
                    std::vector<TimerArray> runs;
                    for (int i = 0; i < kRuns; ++i) {
                        LprResult r;
                        TimerArray t;
                        REQUIRE(model.predict(img, &r, &t));
                        runs.push_back(t);
                    }
                    report("lpr-rec plate_recognition onnx", runs);
                }
            }
        }
    }
}

TEST_CASE("Benchmark OCR single models", "[all_models][benchmark]") {
    auto img = load_img("test_ocr.png");
    auto dict = ocr_dict();
    if (img.empty() || dict.empty()) return;

    // det
    auto det = find_ocr_model("det", ".onnx");
    if (has_file(det)) {
        RuntimeOption opt; opt.use_cpu();
        ocr::DBDetector model(det.string(), opt);
        if (model.is_initialized()) {
            constexpr int kRuns = 20;
            std::vector<TimerArray> runs;
            for (int i = 0; i < kRuns; ++i) {
                std::vector<std::array<int, 8>> boxes;
                TimerArray t;
                REQUIRE(model.predict(img, &boxes, &t));
                if (boxes.empty()) { runs.clear(); break; }
                runs.push_back(t);
            }
            report("ocr-det ppocrv4 onnx", runs);
        }
    }
    // rec
    auto rec = find_ocr_model("rec", ".onnx");
    if (has_file(rec)) {
        RuntimeOption opt; opt.use_cpu();
        ocr::Recognizer model(rec.string(), dict.string(), opt);
        if (model.is_initialized()) {
            constexpr int kRuns = 20;
            std::vector<TimerArray> runs;
            for (int i = 0; i < kRuns; ++i) {
                std::string text; float score = 0;
                TimerArray t;
                REQUIRE(model.predict(img, &text, &score, &t));
                if (text.empty()) { runs.clear(); break; }
                runs.push_back(t);
            }
            report("ocr-rec ppocrv4 onnx", runs);
        }
    }
    // cls
    auto cls = find_ocr_model("cls", ".onnx");
    if (has_file(cls)) {
        RuntimeOption opt; opt.use_cpu();
        ocr::Classifier model(cls.string(), opt);
        if (model.is_initialized()) {
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
                std::cout << "[bench] ocr-cls ppocrv4 onnx | total=" << sum / times.size()
                          << "ms (n=" << times.size() << ")" << std::endl;
            }
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
            auto mp = bench_data_dir() / subdir / be / "insightface" / "buffalo_l" / c.file;
            if (!has_file(mp)) continue;
            RuntimeOption opt;
            opt.use_cpu();
            std::vector<TimerArray> runs;
            constexpr int kRuns = 10;
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
    auto det = bench_data_dir() / "test_models" / "onnx" / "yolov5plate.onnx";
    auto rec = bench_data_dir() / "test_models" / "onnx" / "plate_recognition_color.onnx";
    if (!has_file(det) || !has_file(rec)) return;
    RuntimeOption opt; opt.use_cpu();
    lpr::LprPipeline model(det.string(), rec.string(), opt);
    if (!model.is_initialized()) return;
    auto img = load_img("test_lpr_pipeline.jpg");
    if (img.empty()) return;
    constexpr int kRuns = 20;
    std::vector<TimerArray> runs;
    for (int i = 0; i < kRuns; ++i) {
        std::vector<LprResult> r;
        TimerArray t;
        REQUIRE(model.predict(img, &r, &t));
        runs.push_back(t);
    }
    report("pipeline lpr (det+rec) onnx", runs);
}

TEST_CASE("Benchmark face recognition pipeline", "[pipeline][benchmark]") {
    auto det = bench_data_dir() / "test_models" / "onnx" / "face" / "scrfd_2.5g_bnkps_shape640x640.onnx";
    auto rec = bench_data_dir() / "test_models" / "onnx" / "face" / "face_recognizer.onnx";
    if (!has_file(det) || !has_file(rec)) return;
    RuntimeOption opt; opt.use_cpu();
    face::FaceRecognizerPipeline model(det.string(), rec.string(), opt);
    if (!model.is_initialized()) return;
    auto img = load_img("test_face_detection.jpg");
    if (img.empty()) return;
    constexpr int kRuns = 20;
    std::vector<TimerArray> runs;
    for (int i = 0; i < kRuns; ++i) {
        std::vector<FaceRecognitionResult> r;
        TimerArray t;
        REQUIRE(model.predict(img, &r, &t));
        if (r.empty()) { runs.clear(); break; }
        runs.push_back(t);
    }
    report("pipeline face-rec (scrfd+rec) onnx", runs);
}

TEST_CASE("Benchmark face anti-spoof pipeline", "[pipeline][benchmark]") {
    auto det = bench_data_dir() / "test_models" / "onnx" / "face" / "scrfd_2.5g_bnkps_shape640x640.onnx";
    auto first = bench_data_dir() / "test_models" / "onnx" / "face" / "fas_first.onnx";
    auto second = bench_data_dir() / "test_models" / "onnx" / "face" / "fas_second.onnx";
    if (!has_file(det) || !has_file(first) || !has_file(second)) return;
    RuntimeOption opt; opt.use_cpu();
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
    auto det = bench_data_dir() / "test_models" / "onnx" / "zhgd_det.onnx";
    auto ml = bench_data_dir() / "test_models" / "onnx" / "zhgd_ml.onnx";
    if (!has_file(det) || !has_file(ml)) return;
    RuntimeOption opt; opt.use_cpu();
    pipeline::PedestrianAttribute model(det.string(), ml.string(), opt);
    if (!model.is_initialized()) return;
    model.set_det_input_size({1280, 1280});
    model.set_cls_input_size({192, 256});
    auto img = load_img("test_pedestrian_attribute.jpg");
    if (img.empty()) return;
    constexpr int kRuns = 20;
    std::vector<TimerArray> runs;
    for (int i = 0; i < kRuns; ++i) {
        std::vector<AttributeResult> r;
        TimerArray t;
        REQUIRE(model.predict(img, &r, &t));
        if (r.empty()) { runs.clear(); break; }
        runs.push_back(t);
    }
    report("pipeline pedestrian-attr (det+mlc) onnx", runs);
}

TEST_CASE("Benchmark OCR pipeline", "[pipeline][benchmark]") {
    auto det = find_ocr_model("det", ".onnx");
    auto cls = find_ocr_model("cls", ".onnx");
    auto rec = find_ocr_model("rec", ".onnx");
    auto dict = ocr_dict();
    if (!has_file(det) || !has_file(cls) || !has_file(rec) || dict.empty()) return;
    RuntimeOption opt; opt.use_cpu();
    ocr::PaddleOCR model(det.string(), cls.string(), rec.string(), dict.string(), opt);
    if (!model.is_initialized()) return;
    auto img = load_img("test_ocr.png");
    if (img.empty()) return;
    constexpr int kRuns = 20;
    std::vector<TimerArray> runs;
    for (int i = 0; i < kRuns; ++i) {
        OCRResult r;
        TimerArray t;
        REQUIRE(model.predict(img, &r, &t));
        if (r.boxes.empty()) { runs.clear(); break; }
        runs.push_back(t);
    }
    report("pipeline ocr (det+cls+rec) onnx", runs);
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
