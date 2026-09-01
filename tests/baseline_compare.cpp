// 回归基线对比测试（迁移到 yolo26n 家族 + ppocrv6_tiny）。
//
// 结构：
//  - ORT 端到端：与 `tests/baselines/ort/<模型名>.<ext>.<type>.json` 严格一致（require_no_diff）。
//  - MNN / TRT：与自身基线（require_no_diff）+ ORT 基线（warn_diff）比对。
//  - Sophgo：在设备上运行（本机未编译 Sophgo 后端时 is_initialized()==false 自动跳过）。
//
// 基线的生成：tests/baseline_collect.cpp（--model/--image/--out/--type/--backend/--family）。
// 注意：yolo26n 自带 end2end NMS，不再有 _nms / _without_nms / pre-raw 变体。

#include <catch2/catch_test_macros.hpp>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include <array>
#include "baseline_utils.h"
#include "csrc/vision.h"

namespace fs = std::filesystem;
using namespace modeldeploy;
using namespace modeldeploy::vision;
using namespace modeldeploy::vision::baseline;
using namespace modeldeploy::vision::detection;
using namespace modeldeploy::vision::classification;
using namespace modeldeploy::vision::face;
using namespace modeldeploy::vision::ocr;

static fs::path get_test_data() {
    const char* env = std::getenv("TEST_DATA_DIR");
    if (env && *env) return fs::path(env) / "test_data";
    return fs::current_path() / "test_data";
}

static fs::path baseline_dir(const std::string& backend) {
    return get_test_data().parent_path() / "tests" / "baselines" / backend;
}

static fs::path model_path(const std::string& rel, const std::string& backend = "onnx") {
    return get_test_data() / "test_models" / backend / rel;
}

static fs::path image_path(const std::string& name) {
    return get_test_data() / "test_images" / name;
}

static json load_json(const fs::path& p) {
    std::ifstream f(p);
    json j;
    f >> j;
    return j;
}

static RuntimeOption cpu_option() {
    RuntimeOption opt;
    opt.use_cpu();
    opt.set_cpu_thread_num(4);
    return opt;
}

static RuntimeOption trt_option() {
    RuntimeOption opt;
    opt.use_gpu(0);
    return opt;
}

static void require_no_diff(const std::vector<std::string>& diffs) {
    for (const auto& d : diffs) FAIL_CHECK(d);
}

static void warn_diff(const std::vector<std::string>& diffs, const char* ctx) {
    for (const auto& d : diffs) {
        std::cerr << "[WARN] " << ctx << ": " << d << std::endl;
    }
}

// ---- 自基线（require_no_diff）+ ORT 基线（warn_diff）的双文件比对 ----
static void compare_det_files(const std::vector<DetectionResult>& res,
                              const fs::path& self, const fs::path& ort, const char* ctx) {
    if (fs::exists(self)) require_no_diff(compare_detection(load_json(self)["results"], res));
    if (fs::exists(ort))  warn_diff(compare_detection(load_json(ort)["results"], res), ctx);
}
static void compare_seg_files(const std::vector<InstanceSegResult>& res,
                              const fs::path& self, const fs::path& ort, const char* ctx) {
    if (fs::exists(self)) require_no_diff(compare_seg(load_json(self)["results"], res));
    if (fs::exists(ort))  warn_diff(compare_seg(load_json(ort)["results"], res), ctx);
}
static void compare_pose_files(const std::vector<KeyPointsResult>& res,
                               const fs::path& self, const fs::path& ort, const char* ctx) {
    if (fs::exists(self)) require_no_diff(compare_pose(load_json(self)["results"], res));
    if (fs::exists(ort))  warn_diff(compare_pose(load_json(ort)["results"], res), ctx);
}
static void compare_obb_files(const std::vector<ObbResult>& res,
                              const fs::path& self, const fs::path& ort, const char* ctx) {
    if (fs::exists(self)) require_no_diff(compare_obb(load_json(self)["results"], res));
    if (fs::exists(ort))  warn_diff(compare_obb(load_json(ort)["results"], res), ctx);
}
static void compare_cls_files(const ClassifyResult& r,
                              const fs::path& self, const fs::path& ort, const char* ctx) {
    if (fs::exists(self)) require_no_diff(compare_cls(load_json(self)["results"], r));
    if (fs::exists(ort))  warn_diff(compare_cls(load_json(ort)["results"], r), ctx);
}

// ==================== ORT 端到端（严格比对）====================
TEST_CASE("yolo26n detection (ORT)", "[regression]") {
    auto modelfile = model_path("yolo26n/yolo26n.onnx");
    if (!fs::exists(modelfile)) return;
    auto imgf = image_path("test_detection0.jpg");
    if (!fs::exists(imgf)) return;
    auto base = baseline_dir("ort") / "yolo26n.onnx.det.json";
    if (!fs::exists(base)) return;

    UltralyticsDet model(modelfile.string(), cpu_option());
    REQUIRE(model.is_initialized());
    auto img = ImageData::imread(imgf.string());
    REQUIRE_FALSE(img.empty());

    std::vector<DetectionResult> results;
    REQUIRE(model.predict(img, &results, nullptr));
    require_no_diff(compare_detection(load_json(base)["results"], results));
}

// 裁剪/预处理链路护栏：运行 yolo26n 为检测输入的预处理（含 crop/letterbox/resize/normalize），
// 其产出输入张量须与 `tests/baselines/ort/yolo26n.onnx.pre.json` 严格一致。
// 基线的生成：baseline_collect --model yolo26n/yolo26n.onnx --image test_detection0.jpg \
//   --out tests/baselines/ort --type pre --family det --backend ort
TEST_CASE("yolo26n detection pre-tensor (ORT) guard", "[regression]") {
    auto modelfile = model_path("yolo26n/yolo26n.onnx");
    if (!fs::exists(modelfile)) return;
    auto imgf = image_path("test_detection0.jpg");
    if (!fs::exists(imgf)) return;
    auto base = baseline_dir("ort") / "yolo26n.onnx.pre.json";
    if (!fs::exists(base)) return;

    UltralyticsDet model(modelfile.string(), cpu_option());
    REQUIRE(model.is_initialized());
    auto img = ImageData::imread(imgf.string());
    REQUIRE_FALSE(img.empty());

    std::vector<Tensor> inputs;
    std::vector<LetterBoxRecord> recs;
    REQUIRE(model.get_preprocessor().run({img}, &inputs, &recs));
    REQUIRE_FALSE(inputs.empty());
    require_no_diff(compare_tensor(load_json(base)["tensor"], inputs[0]));
}

TEST_CASE("yolo26n-seg segmentation (ORT)", "[regression]") {
    auto modelfile = model_path("yolo26n/yolo26n-seg.onnx");
    if (!fs::exists(modelfile)) return;
    auto imgf = image_path("test_person.jpg");
    if (!fs::exists(imgf)) return;
    auto base = baseline_dir("ort") / "yolo26n-seg.onnx.seg.json";
    if (!fs::exists(base)) return;

    UltralyticsSeg model(modelfile.string(), cpu_option());
    REQUIRE(model.is_initialized());
    auto img = ImageData::imread(imgf.string());
    REQUIRE_FALSE(img.empty());

    std::vector<InstanceSegResult> results;
    REQUIRE(model.predict(img, &results, nullptr));
    require_no_diff(compare_seg(load_json(base)["results"], results));
}

TEST_CASE("yolo26n-pose pose estimation (ORT)", "[regression]") {
    auto modelfile = model_path("yolo26n/yolo26n-pose.onnx");
    if (!fs::exists(modelfile)) return;
    auto imgf = image_path("test_person.jpg");
    if (!fs::exists(imgf)) return;
    auto base = baseline_dir("ort") / "yolo26n-pose.onnx.pose.json";
    if (!fs::exists(base)) return;

    UltralyticsPose model(modelfile.string(), cpu_option());
    REQUIRE(model.is_initialized());
    auto img = ImageData::imread(imgf.string());
    REQUIRE_FALSE(img.empty());

    std::vector<KeyPointsResult> results;
    REQUIRE(model.predict(img, &results, nullptr));
    require_no_diff(compare_pose(load_json(base)["results"], results));
}

TEST_CASE("yolo26n-obb obb detection (ORT)", "[regression]") {
    auto modelfile = model_path("yolo26n/yolo26n-obb.onnx");
    if (!fs::exists(modelfile)) return;
    auto imgf = image_path("test_obb1.jpg");
    if (!fs::exists(imgf)) return;
    auto base = baseline_dir("ort") / "yolo26n-obb.onnx.obb.json";
    if (!fs::exists(base)) return;

    UltralyticsObb model(modelfile.string(), cpu_option());
    REQUIRE(model.is_initialized());
    model.get_preprocessor().set_size({1024, 1024});
    auto img = ImageData::imread(imgf.string());
    REQUIRE_FALSE(img.empty());

    std::vector<ObbResult> results;
    REQUIRE(model.predict(img, &results, nullptr));
    require_no_diff(compare_obb(load_json(base)["results"], results));
}

TEST_CASE("yolo26n-cls classification (ORT)", "[regression]") {
    auto modelfile = model_path("yolo26n/yolo26n-cls.onnx");
    if (!fs::exists(modelfile)) return;
    auto imgf = image_path("test_person.jpg");
    if (!fs::exists(imgf)) return;
    auto base = baseline_dir("ort") / "yolo26n-cls.onnx.cls.json";
    if (!fs::exists(base)) return;

    Classification model(modelfile.string(), cpu_option());
    REQUIRE(model.is_initialized());
    auto img = ImageData::imread(imgf.string());
    REQUIRE_FALSE(img.empty());

    ClassifyResult result;
    REQUIRE(model.predict(img, &result));
    require_no_diff(compare_cls(load_json(base)["results"], result));
}

TEST_CASE("scrfd face detection (ORT)", "[regression]") {
    auto modelfile = model_path("seetaface/scrfd_2.5g_bnkps_shape640x640.onnx");
    if (!fs::exists(modelfile)) return;
    auto imgf = image_path("test_face_detection.jpg");
    if (!fs::exists(imgf)) return;
    auto base = baseline_dir("ort") / "scrfd_2.5g_bnkps_shape640x640.onnx.face_det.json";
    if (!fs::exists(base)) return;

    Scrfd model(modelfile.string(), cpu_option());
    REQUIRE(model.is_initialized());
    auto img = ImageData::imread(imgf.string());
    REQUIRE_FALSE(img.empty());

    std::vector<KeyPointsResult> results;
    REQUIRE(model.predict(img, &results, nullptr));
    require_no_diff(compare_pose(load_json(base)["results"], results));
}

// ==================== OCR（ppocrv6_tiny）====================
TEST_CASE("ppocrv6_tiny det (ORT)", "[regression]") {
    auto modelfile = model_path("ocr/ppocrv6_tiny/det_infer.onnx");
    if (!fs::exists(modelfile)) return;
    auto imgf = image_path("test_ocr.png");
    if (!fs::exists(imgf)) return;
    auto base = baseline_dir("ort") / "det_infer.onnx.ocr_det.json";
    if (!fs::exists(base)) return;

    DBDetector model(modelfile.string(), cpu_option());
    REQUIRE(model.is_initialized());
    auto img = ImageData::imread(imgf.string());
    REQUIRE_FALSE(img.empty());

    std::vector<std::array<int, 8>> boxes;
    REQUIRE(model.predict(img, &boxes, nullptr));
    require_no_diff(compare_ocr_det(load_json(base)["results"], boxes));
}

TEST_CASE("ppocrv6_tiny rec (ORT)", "[regression]") {
    auto modelfile = model_path("ocr/ppocrv6_tiny/rec_infer.onnx");
    if (!fs::exists(modelfile)) return;
    // rec 模型输入为单行文本裁剪图，整图（test_ocr.png）会因输出空容器崩溃/失败
    auto imgf = image_path("test_ocr_recognition.jpg");
    if (!fs::exists(imgf)) return;
    auto base = baseline_dir("ort") / "rec_infer.onnx.ocr_rec.json";
    if (!fs::exists(base)) return;
    auto dict = get_test_data() / "ppocrv6_tiny_dict.txt";
    if (!fs::exists(dict)) return;

    Recognizer model(modelfile.string(), dict.string(), cpu_option());
    REQUIRE(model.is_initialized());
    auto img = ImageData::imread(imgf.string());
    REQUIRE_FALSE(img.empty());

    std::string text;
    float score = 0;
    REQUIRE(model.predict(img, &text, &score, nullptr));
    require_no_diff(compare_ocr_rec(load_json(base)["results"], text, score));
}

TEST_CASE("ppocrv6_tiny cls (ORT)", "[regression]") {
    auto modelfile = model_path("ocr/ppocrv6_tiny/cls_infer.onnx");
    if (!fs::exists(modelfile)) return;
    auto imgf = image_path("test_ocr.png");
    if (!fs::exists(imgf)) return;
    auto base = baseline_dir("ort") / "cls_infer.onnx.ocr_cls.json";
    if (!fs::exists(base)) return;

    Classifier model(modelfile.string(), cpu_option());
    REQUIRE(model.is_initialized());
    auto img = ImageData::imread(imgf.string());
    REQUIRE_FALSE(img.empty());

    int32_t label = -1;
    float score = 0;
    REQUIRE(model.predict(img, &label, &score));
    require_no_diff(compare_ocr_cls(load_json(base)["results"], label, score));
}

// ==================== MNN ====================
#ifdef ENABLE_MNN
TEST_CASE("yolo26n detection MNN", "[regression][backend:mnn]") {
    auto modelfile = model_path("yolo26n/yolo26n.mnn", "mnn");
    if (!fs::exists(modelfile)) return;
    auto imgf = image_path("test_detection0.jpg");
    if (!fs::exists(imgf)) return;

    UltralyticsDet model(modelfile.string(), cpu_option());
    REQUIRE(model.is_initialized());
    auto img = ImageData::imread(imgf.string());
    REQUIRE_FALSE(img.empty());

    std::vector<DetectionResult> results;
    REQUIRE(model.predict(img, &results, nullptr));
    compare_det_files(results, baseline_dir("mnn") / "yolo26n.mnn.det.json",
                      baseline_dir("ort") / "yolo26n.onnx.det.json", "yolo26n.mnn vs ORT");
}

TEST_CASE("yolo26n-cls classification MNN", "[regression][backend:mnn]") {
    auto modelfile = model_path("yolo26n/yolo26n-cls.mnn", "mnn");
    if (!fs::exists(modelfile)) return;
    auto imgf = image_path("test_person.jpg");
    if (!fs::exists(imgf)) return;

    Classification model(modelfile.string(), cpu_option());
    REQUIRE(model.is_initialized());
    auto img = ImageData::imread(imgf.string());
    REQUIRE_FALSE(img.empty());

    ClassifyResult result;
    REQUIRE(model.predict(img, &result));
    compare_cls_files(result, baseline_dir("mnn") / "yolo26n-cls.mnn.cls.json",
                      baseline_dir("ort") / "yolo26n-cls.onnx.cls.json", "yolo26n-cls.mnn vs ORT");
}

TEST_CASE("yolo26n-obb obb detection MNN", "[regression][backend:mnn]") {
    auto modelfile = model_path("yolo26n/yolo26n-obb.mnn", "mnn");
    if (!fs::exists(modelfile)) return;
    auto imgf = image_path("test_obb1.jpg");
    if (!fs::exists(imgf)) return;

    UltralyticsObb model(modelfile.string(), cpu_option());
    REQUIRE(model.is_initialized());
    model.get_preprocessor().set_size({1024, 1024});
    auto img = ImageData::imread(imgf.string());
    REQUIRE_FALSE(img.empty());

    std::vector<ObbResult> results;
    REQUIRE(model.predict(img, &results, nullptr));
    compare_obb_files(results, baseline_dir("mnn") / "yolo26n-obb.mnn.obb.json",
                      baseline_dir("ort") / "yolo26n-obb.onnx.obb.json", "yolo26n-obb.mnn vs ORT");
}

TEST_CASE("yolo26n-pose pose estimation MNN", "[regression][backend:mnn]") {
    auto modelfile = model_path("yolo26n/yolo26n-pose.mnn", "mnn");
    if (!fs::exists(modelfile)) return;
    auto imgf = image_path("test_person.jpg");
    if (!fs::exists(imgf)) return;

    UltralyticsPose model(modelfile.string(), cpu_option());
    REQUIRE(model.is_initialized());
    auto img = ImageData::imread(imgf.string());
    REQUIRE_FALSE(img.empty());

    std::vector<KeyPointsResult> results;
    REQUIRE(model.predict(img, &results, nullptr));
    compare_pose_files(results, baseline_dir("mnn") / "yolo26n-pose.mnn.pose.json",
                       baseline_dir("ort") / "yolo26n-pose.onnx.pose.json", "yolo26n-pose.mnn vs ORT");
}

TEST_CASE("yolo26n-seg segmentation MNN", "[regression][backend:mnn]") {
    auto modelfile = model_path("yolo26n/yolo26n-seg.mnn", "mnn");
    if (!fs::exists(modelfile)) return;
    auto imgf = image_path("test_person.jpg");
    if (!fs::exists(imgf)) return;

    UltralyticsSeg model(modelfile.string(), cpu_option());
    REQUIRE(model.is_initialized());
    auto img = ImageData::imread(imgf.string());
    REQUIRE_FALSE(img.empty());

    std::vector<InstanceSegResult> results;
    REQUIRE(model.predict(img, &results, nullptr));
    compare_seg_files(results, baseline_dir("mnn") / "yolo26n-seg.mnn.seg.json",
                      baseline_dir("ort") / "yolo26n-seg.onnx.seg.json", "yolo26n-seg.mnn vs ORT");
}
#endif // ENABLE_MNN

// ==================== TRT ====================
#ifdef ENABLE_TRT
TEST_CASE("yolo26n detection TRT", "[regression][backend:trt]") {
    auto modelfile = model_path("yolo26n/yolo26n.engine", "trt");
    if (!fs::exists(modelfile)) return;
    auto imgf = image_path("test_detection0.jpg");
    if (!fs::exists(imgf)) return;

    UltralyticsDet model(modelfile.string(), trt_option());
    if (!model.is_initialized()) return;   // TRT backend 未构建 -> 跳过
    auto img = ImageData::imread(imgf.string());
    REQUIRE_FALSE(img.empty());

    std::vector<DetectionResult> results;
    REQUIRE(model.predict(img, &results, nullptr));
    compare_det_files(results, baseline_dir("trt") / "yolo26n.engine.det.json",
                      baseline_dir("ort") / "yolo26n.onnx.det.json", "yolo26n.engine vs ORT");
}

TEST_CASE("yolo26n-cls classification TRT", "[regression][backend:trt]") {
    auto modelfile = model_path("yolo26n/yolo26n-cls.engine", "trt");
    if (!fs::exists(modelfile)) return;
    auto imgf = image_path("test_person.jpg");
    if (!fs::exists(imgf)) return;

    Classification model(modelfile.string(), trt_option());
    if (!model.is_initialized()) return;   // TRT backend 未构建 -> 跳过
    auto img = ImageData::imread(imgf.string());
    REQUIRE_FALSE(img.empty());

    ClassifyResult result;
    REQUIRE(model.predict(img, &result));
    compare_cls_files(result, baseline_dir("trt") / "yolo26n-cls.engine.cls.json",
                      baseline_dir("ort") / "yolo26n-cls.onnx.cls.json", "yolo26n-cls.engine vs ORT");
}

TEST_CASE("yolo26n-obb obb detection TRT", "[regression][backend:trt]") {
    auto modelfile = model_path("yolo26n/yolo26n-obb.engine", "trt");
    if (!fs::exists(modelfile)) return;
    auto imgf = image_path("test_obb1.jpg");
    if (!fs::exists(imgf)) return;

    UltralyticsObb model(modelfile.string(), trt_option());
    model.get_preprocessor().set_size({1024, 1024});
    if (!model.is_initialized()) return;   // TRT backend 未构建 -> 跳过
    auto img = ImageData::imread(imgf.string());
    REQUIRE_FALSE(img.empty());

    std::vector<ObbResult> results;
    REQUIRE(model.predict(img, &results, nullptr));
    compare_obb_files(results, baseline_dir("trt") / "yolo26n-obb.engine.obb.json",
                      baseline_dir("ort") / "yolo26n-obb.onnx.obb.json", "yolo26n-obb.engine vs ORT");
}

TEST_CASE("yolo26n-pose pose estimation TRT", "[regression][backend:trt]") {
    auto modelfile = model_path("yolo26n/yolo26n-pose.engine", "trt");
    if (!fs::exists(modelfile)) return;
    auto imgf = image_path("test_person.jpg");
    if (!fs::exists(imgf)) return;

    UltralyticsPose model(modelfile.string(), trt_option());
    if (!model.is_initialized()) return;   // TRT backend 未构建 -> 跳过
    auto img = ImageData::imread(imgf.string());
    REQUIRE_FALSE(img.empty());

    std::vector<KeyPointsResult> results;
    REQUIRE(model.predict(img, &results, nullptr));
    compare_pose_files(results, baseline_dir("trt") / "yolo26n-pose.engine.pose.json",
                       baseline_dir("ort") / "yolo26n-pose.onnx.pose.json", "yolo26n-pose.engine vs ORT");
}

TEST_CASE("yolo26n-seg segmentation TRT", "[regression][backend:trt]") {
    auto modelfile = model_path("yolo26n/yolo26n-seg.engine", "trt");
    if (!fs::exists(modelfile)) return;
    auto imgf = image_path("test_person.jpg");
    if (!fs::exists(imgf)) return;

    UltralyticsSeg model(modelfile.string(), trt_option());
    if (!model.is_initialized()) return;   // TRT backend 未构建 -> 跳过
    auto img = ImageData::imread(imgf.string());
    REQUIRE_FALSE(img.empty());

    std::vector<InstanceSegResult> results;
    REQUIRE(model.predict(img, &results, nullptr));
    compare_seg_files(results, baseline_dir("trt") / "yolo26n-seg.engine.seg.json",
                      baseline_dir("ort") / "yolo26n-seg.onnx.seg.json", "yolo26n-seg.engine vs ORT");
}
#endif // ENABLE_TRT

// ==================== Sophgo（int8，设备运行；本机缺后端自动跳过）====================
TEST_CASE("yolo26n detection Sophgo", "[backend:sophgo]") {
    auto modelfile = model_path("yolo26n/yolo26n-int8.bmodel", "sophgo");
    if (!fs::exists(modelfile)) return;
    auto imgf = image_path("test_detection0.png");
    if (!fs::exists(imgf)) return;

    UltralyticsDet model(modelfile.string(), cpu_option());
    if (!model.is_initialized()) return;   // Sophgo backend 未构建 -> 跳过
    auto img = ImageData::imread(imgf.string());
    REQUIRE_FALSE(img.empty());

    std::vector<DetectionResult> results;
    REQUIRE(model.predict(img, &results, nullptr));
    compare_det_files(results, baseline_dir("sophgo") / "yolo26n-int8.bmodel.det.json",
                      baseline_dir("ort") / "yolo26n.onnx.det.json", "yolo26n-int8.bmodel vs ORT");
}

TEST_CASE("yolo26n-cls classification Sophgo", "[backend:sophgo]") {
    auto modelfile = model_path("yolo26n/yolo26n-cls-int8.bmodel", "sophgo");
    if (!fs::exists(modelfile)) return;
    auto imgf = image_path("test_person.png");
    if (!fs::exists(imgf)) return;

    Classification model(modelfile.string(), cpu_option());
    if (!model.is_initialized()) return;   // Sophgo backend 未构建 -> 跳过
    auto img = ImageData::imread(imgf.string());
    REQUIRE_FALSE(img.empty());

    ClassifyResult result;
    REQUIRE(model.predict(img, &result));
    compare_cls_files(result, baseline_dir("sophgo") / "yolo26n-cls-int8.bmodel.cls.json",
                      baseline_dir("ort") / "yolo26n-cls.onnx.cls.json", "yolo26n-cls-int8.bmodel vs ORT");
}

TEST_CASE("yolo26n-obb obb detection Sophgo", "[backend:sophgo]") {
    auto modelfile = model_path("yolo26n/yolo26n-obb-int8.bmodel", "sophgo");
    if (!fs::exists(modelfile)) return;
    auto imgf = image_path("test_obb1.png");
    if (!fs::exists(imgf)) return;

    UltralyticsObb model(modelfile.string(), cpu_option());
    if (!model.is_initialized()) return;   // Sophgo backend 未构建 -> 跳过
    model.get_preprocessor().set_size({1024, 1024});
    auto img = ImageData::imread(imgf.string());
    REQUIRE_FALSE(img.empty());

    std::vector<ObbResult> results;
    REQUIRE(model.predict(img, &results, nullptr));
    compare_obb_files(results, baseline_dir("sophgo") / "yolo26n-obb-int8.bmodel.obb.json",
                      baseline_dir("ort") / "yolo26n-obb.onnx.obb.json", "yolo26n-obb-int8.bmodel vs ORT");
}

TEST_CASE("yolo26n-pose pose estimation Sophgo", "[backend:sophgo]") {
    auto modelfile = model_path("yolo26n/yolo26n-pose-int8.bmodel", "sophgo");
    if (!fs::exists(modelfile)) return;
    auto imgf = image_path("test_person.png");
    if (!fs::exists(imgf)) return;

    UltralyticsPose model(modelfile.string(), cpu_option());
    if (!model.is_initialized()) return;   // Sophgo backend 未构建 -> 跳过
    auto img = ImageData::imread(imgf.string());
    REQUIRE_FALSE(img.empty());

    std::vector<KeyPointsResult> results;
    REQUIRE(model.predict(img, &results, nullptr));
    compare_pose_files(results, baseline_dir("sophgo") / "yolo26n-pose-int8.bmodel.pose.json",
                       baseline_dir("ort") / "yolo26n-pose.onnx.pose.json", "yolo26n-pose-int8.bmodel vs ORT");
}

TEST_CASE("yolo26n-seg segmentation Sophgo", "[backend:sophgo]") {
    auto modelfile = model_path("yolo26n/yolo26n-seg-int8.bmodel", "sophgo");
    if (!fs::exists(modelfile)) return;
    auto imgf = image_path("test_person.png");
    if (!fs::exists(imgf)) return;

    UltralyticsSeg model(modelfile.string(), cpu_option());
    if (!model.is_initialized()) return;   // Sophgo backend 未构建 -> 跳过
    auto img = ImageData::imread(imgf.string());
    REQUIRE_FALSE(img.empty());

    std::vector<InstanceSegResult> results;
    REQUIRE(model.predict(img, &results, nullptr));
    compare_seg_files(results, baseline_dir("sophgo") / "yolo26n-seg-int8.bmodel.seg.json",
                      baseline_dir("ort") / "yolo26n-seg.onnx.seg.json", "yolo26n-seg-int8.bmodel vs ORT");
}
