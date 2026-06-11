#include <catch2/catch_test_macros.hpp>
#include <catch2/benchmark/catch_benchmark.hpp>
#include <opencv2/core/mat.hpp>
#include <string>
#include <vector>
#include <cstring>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>

#include "capi/common/md_types.h"
#include "capi/common/md_decl.h"
#include "capi/utils/md_image_capi.h"
#include "capi/vision/detection/detection_capi.h"
#include "capi/vision/classification/classification_capi.h"
#include "capi/vision/obb/obb_capi.h"
#include "capi/vision/pose/pose_capi.h"
#include "capi/vision/iseg/instance_seg_capi.h"
#include "core/md_log.h"

using namespace modeldeploy;

// RAII 日志静音器：抑制模型加载时的 INFO 日志和 std::cout 输出
struct LogSilencer {
    std::streambuf* old_buf;
    std::ofstream null_stream;
    LogSilencer() : null_stream("nul") {
        old_buf = std::cout.rdbuf(null_stream.rdbuf());
        MD_SET_LOG_LEVEL(LogLevel::MD_LOG_W);
    }
    ~LogSilencer() {
        std::cout.rdbuf(old_buf);
        MD_SET_LOG_LEVEL(LogLevel::MD_LOG_I);
    }
};

namespace fs = std::filesystem;

// ==================== Path helpers ====================
static fs::path model_dir() {
    const char* env = std::getenv("TEST_DATA_DIR");
    return ((env && *env) ? fs::path(env) : fs::current_path()) / "test_data" / "test_models";
}
static fs::path img_file() {
    const char* env = std::getenv("TEST_DATA_DIR");
    return ((env && *env) ? fs::path(env) : fs::current_path()) / "test_data" / "test_images" / "test_person.jpg";
}

// ==================== RuntimeOption builder ====================
static MDRuntimeOption make_opt(int backend, int device, int threads) {
    MDRuntimeOption opt;
    memset(&opt, 0, sizeof(opt));
    opt.cpu_thread_num = threads;
    opt.device = static_cast<MDDevice>(device);
    opt.backend = static_cast<MDBackend>(backend);
    opt.enable_trt = 0;
    opt.trt_min_shape = "";
    opt.trt_opt_shape = "";
    opt.trt_max_shape = "";
    opt.trt_engine_cache_path = "";
    opt.password = "";
    opt.ort_log_severity = 3;
    return opt;
}

static const char* backend_name(int b) {
    switch (b) { case 0: return "ORT"; case 1: return "MNN"; case 2: return "TRT"; default: return "?"; }
}
static const char* device_name(int d) {
    switch (d) { case 0: return "CPU"; case 1: return "GPU"; default: return "?"; }
}

// ==================== Image operation bench (md_clone_image) ====================
TEST_CASE("Image clone 640x480", "[imgproc][benchmark]") {
    auto img = md_read_image(img_file().string().c_str());
    if (!img.data) return;
    BENCHMARK("clone 640x480") { return md_clone_image(&img); };
    md_free_image(&img);
}

// ==================== Detection ====================
struct BenchCfg {
    const char* tag;
    const char* model_rel;
    int backend;
    int device;
    int threads;
};

static bool file_exists(const char* path) { return fs::exists(path); }

static void bench_det(const BenchCfg& c) {
    auto mp = model_dir() / c.model_rel;
    if (!file_exists(mp.string().c_str())) return;
    LogSilencer _silence;
    MDModel m; memset(&m, 0, sizeof(m));
    auto opt = make_opt(c.backend, c.device, c.threads);
    if (md_create_detection_model(&m, mp.string().c_str(), &opt) != 0) return;
    auto img = md_read_image(img_file().string().c_str());
    if (!img.data) { md_free_detection_model(&m); return; }
    BENCHMARK(c.tag) {
        MDDetectionResults r; memset(&r, 0, sizeof(r));
        md_detection_predict(&m, &img, &r);
        md_free_detection_result(&r);
    };
    md_free_image(&img);
    md_free_detection_model(&m);
}

TEST_CASE("Detection inference", "[model][det][benchmark]") {
    BenchCfg cfgs[16];
    int n = 0;
#ifdef ENABLE_ORT
    cfgs[n++] = {"det/ort/cpu/t1", "yolo11n_nms.onnx", 0, 0, 1};
    cfgs[n++] = {"det/ort/cpu/t4", "yolo11n_nms.onnx", 0, 0, 4};
    cfgs[n++] = {"det/ort/cpu/t8", "yolo11n_nms.onnx", 0, 0, 8};
    cfgs[n++] = {"det/ort/gpu",    "yolo11n_nms.onnx", 0, 1, 4};
#endif
#ifdef ENABLE_MNN
    cfgs[n++] = {"det/mnn/cpu/t4", "yolo11n_nms.mnn", 1, 0, 4};
#endif
#if defined(WITH_GPU) && defined(ENABLE_TRT)
    cfgs[n++] = {"det/trt/gpu",    "yolo11n.engine",  2, 1, 4};
    cfgs[n++] = {"det/trt/onnx",   "yolo11n_nms.onnx",2, 1, 4};
#endif
    for (int i = 0; i < n; ++i) bench_det(cfgs[i]);
}

// ==================== Classification ====================
static void bench_cls(const BenchCfg& c) {
    auto mp = model_dir() / c.model_rel;
    if (!file_exists(mp.string().c_str())) return;
    MDModel m; memset(&m, 0, sizeof(m));
    auto opt = make_opt(c.backend, c.device, c.threads);
    if (md_create_classification_model(&m, mp.string().c_str(), &opt) != 0) return;
    auto img = md_read_image(img_file().string().c_str());
    if (!img.data) { md_free_classification_model(&m); return; }
    BENCHMARK(c.tag) {
        MDClassificationResults r; memset(&r, 0, sizeof(r));
        md_classification_predict(&m, &img, &r);
        md_free_classification_result(&r);
    };
    md_free_image(&img);
    md_free_classification_model(&m);
}

TEST_CASE("Classification inference", "[model][cls][benchmark]") {
    BenchCfg cfgs[4];
    int n = 0;
#ifdef ENABLE_ORT
    cfgs[n++] = {"cls/ort/cpu/t4", "yolo11n-cls.onnx", 0, 0, 4};
    cfgs[n++] = {"cls/ort/gpu",    "yolo11n-cls.onnx", 0, 1, 4};
#endif
    for (int i = 0; i < n; ++i) bench_cls(cfgs[i]);
}

// ==================== OBB ====================
static void bench_obb(const BenchCfg& c) {
    auto mp = model_dir() / c.model_rel;
    if (!file_exists(mp.string().c_str())) return;
    MDModel m; memset(&m, 0, sizeof(m));
    auto opt = make_opt(c.backend, c.device, c.threads);
    if (md_create_obb_model(&m, mp.string().c_str(), &opt) != 0) return;
    auto img = md_read_image(img_file().string().c_str());
    if (!img.data) { md_free_obb_model(&m); return; }
    BENCHMARK(c.tag) {
        MDObbResults r; memset(&r, 0, sizeof(r));
        md_obb_predict(&m, &img, &r);
        md_free_obb_result(&r);
    };
    md_free_image(&img);
    md_free_obb_model(&m);
}

TEST_CASE("OBB inference", "[model][obb][benchmark]") {
    BenchCfg cfgs[4];
    int n = 0;
#ifdef ENABLE_ORT
    cfgs[n++] = {"obb/ort/cpu/t4", "yolo11n-obb_nms.onnx", 0, 0, 4};
    cfgs[n++] = {"obb/ort/gpu",    "yolo11n-obb_nms.onnx", 0, 1, 4};
#endif
    for (int i = 0; i < n; ++i) bench_obb(cfgs[i]);
}

// ==================== Pose ====================
static void bench_pose(const BenchCfg& c) {
    auto mp = model_dir() / c.model_rel;
    if (!file_exists(mp.string().c_str())) return;
    MDModel m; memset(&m, 0, sizeof(m));
    auto opt = make_opt(c.backend, c.device, c.threads);
    if (md_create_keypoint_model(&m, mp.string().c_str(), &opt) != 0) return;
    auto img = md_read_image(img_file().string().c_str());
    if (!img.data) { md_free_keypoint_model(&m); return; }
    BENCHMARK(c.tag) {
        MDKeyPointResults r; memset(&r, 0, sizeof(r));
        md_keypoint_predict(&m, &img, &r);
        md_free_keypoint_result(&r);
    };
    md_free_image(&img);
    md_free_keypoint_model(&m);
}

TEST_CASE("Pose inference", "[model][pose][benchmark]") {
    BenchCfg cfgs[4];
    int n = 0;
#ifdef ENABLE_ORT
    cfgs[n++] = {"pose/ort/cpu/t4", "yolo11n-pose_nms.onnx", 0, 0, 4};
    cfgs[n++] = {"pose/ort/gpu",    "yolo11n-pose_nms.onnx", 0, 1, 4};
#endif
    for (int i = 0; i < n; ++i) bench_pose(cfgs[i]);
}

// ==================== Segmentation ====================
static void bench_seg(const BenchCfg& c) {
    auto mp = model_dir() / c.model_rel;
    if (!file_exists(mp.string().c_str())) return;
    MDModel m; memset(&m, 0, sizeof(m));
    auto opt = make_opt(c.backend, c.device, c.threads);
    if (md_create_instance_seg_model(&m, mp.string().c_str(), &opt) != 0) return;
    auto img = md_read_image(img_file().string().c_str());
    if (!img.data) { md_free_instance_seg_model(&m); return; }
    BENCHMARK(c.tag) {
        MDIsegResults r; memset(&r, 0, sizeof(r));
        md_instance_seg_predict(&m, &img, &r);
        md_free_instance_seg_result(&r);
    };
    md_free_image(&img);
    md_free_instance_seg_model(&m);
}

TEST_CASE("Segmentation inference", "[model][seg][benchmark]") {
    BenchCfg cfgs[4];
    int n = 0;
#ifdef ENABLE_ORT
    cfgs[n++] = {"seg/ort/cpu/t4", "yolo11n-seg_nms.onnx", 0, 0, 4};
    cfgs[n++] = {"seg/ort/gpu",    "yolo11n-seg_nms.onnx", 0, 1, 4};
#endif
    for (int i = 0; i < n; ++i) bench_seg(cfgs[i]);
}

// ==================== Model load time ====================
TEST_CASE("Model load time", "[load][benchmark]") {
    struct LoadTest { const char* name; const char* rel; int backend; int device; int threads; MDStatusCode (*create)(MDModel*, const char*, const MDRuntimeOption*); };
    LoadTest lt[16];
    int n = 0;
#ifdef ENABLE_ORT
    lt[n++] = {"det/ort/cpu", "yolo11n_nms.onnx", 0, 0, 4, md_create_detection_model};
    lt[n++] = {"det/ort/gpu", "yolo11n_nms.onnx", 0, 1, 4, md_create_detection_model};
    lt[n++] = {"cls/ort/cpu", "yolo11n-cls.onnx", 0, 0, 4, md_create_classification_model};
    lt[n++] = {"obb/ort/cpu", "yolo11n-obb_nms.onnx", 0, 0, 4, md_create_obb_model};
    lt[n++] = {"pose/ort/cpu","yolo11n-pose_nms.onnx",0, 0, 4, md_create_keypoint_model};
    lt[n++] = {"seg/ort/cpu", "yolo11n-seg_nms.onnx", 0, 0, 4, md_create_instance_seg_model};
#endif
#ifdef ENABLE_MNN
    lt[n++] = {"det/mnn/cpu", "yolo11n_nms.mnn",  1, 0, 4, md_create_detection_model};
#endif
#if defined(WITH_GPU) && defined(ENABLE_TRT)
    lt[n++] = {"det/trt/gpu", "yolo11n.engine",  2, 1, 4, md_create_detection_model};
#endif
    for (int i = 0; i < n; ++i) {
        auto mp = model_dir() / lt[i].rel;
        if (!file_exists(mp.string().c_str())) continue;
        auto opt = make_opt(lt[i].backend, lt[i].device, lt[i].threads);
        BENCHMARK(lt[i].name) {
            LogSilencer _s;
            MDModel m; memset(&m, 0, sizeof(m));
            lt[i].create(&m, mp.string().c_str(), &opt);
        };
    }
}
