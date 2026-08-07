#include <catch2/catch_test_macros.hpp>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>
#include <array>
#include "baseline_utils.h"
#include "csrc/vision.h"
#include "csrc/vision/common/visualize/visualize.h"

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

static fs::path baseline_dir() { return get_test_data().parent_path() / "tests" / "baselines"; }

static fs::path model_path(const std::string& rel) {
    return get_test_data() / "test_models" / rel;
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

static void require_no_diff(const std::vector<std::string>& diffs) {
    for (const auto& d : diffs) FAIL_CHECK(d);
}

static void check_tensors(const std::vector<fs::path>& files,
                          const std::vector<Tensor>& tensors) {
    REQUIRE(files.size() == tensors.size());
    for (size_t i = 0; i < files.size(); ++i) {
        if (!fs::exists(files[i])) return;
        auto base = load_json(files[i]);
        require_no_diff(compare_tensor(base["tensor"], tensors[i]));
    }
}

template <typename Model>
static void compare_yolo_pre_raw(Model& model, const ImageData& img,
                                 const fs::path& pre_file, const fs::path& raw_file) {
    auto& preproc = model.get_preprocessor();
    preproc.set_size({640, 640});
    std::vector<LetterBoxRecord> lbs;
    std::vector<Tensor> inputs;
    REQUIRE(preproc.run({img}, &inputs, &lbs));
    std::vector<Tensor> outputs;
    REQUIRE(model.infer(inputs, &outputs));
    REQUIRE_FALSE(inputs.empty());
    REQUIRE_FALSE(outputs.empty());
    check_tensors({pre_file, raw_file}, {inputs[0], outputs[0]});
}

TEST_CASE("Regression: yolo11n detection + pre/raw", "[regression]") {
    auto modelfile = model_path("yolo11n.onnx");
    if (!fs::exists(modelfile)) return;
    auto imgf = image_path("test_detection0.jpg");
    if (!fs::exists(imgf)) return;
    auto base_file = baseline_dir() / "yolo11n.onnx.det.json";
    if (!fs::exists(base_file)) return;

    UltralyticsDet model(modelfile.string(), cpu_option());
    if (!model.is_initialized()) return;

    auto img = ImageData::imread(imgf.string());
    if (img.empty()) return;

    std::vector<DetectionResult> results;
    REQUIRE(model.predict(img, &results, nullptr));
    require_no_diff(compare_detection(load_json(base_file)["results"], results));

    compare_yolo_pre_raw(model, img,
                         baseline_dir() / "yolo11n.onnx.pre.json",
                         baseline_dir() / "yolo11n.onnx.raw.json");
}

TEST_CASE("Regression: yolo11n_nms detection", "[regression]") {
    auto modelfile = model_path("yolo11n_nms.onnx");
    if (!fs::exists(modelfile)) return;
    auto imgf = image_path("test_detection0.jpg");
    if (!fs::exists(imgf)) return;
    auto base_file = baseline_dir() / "yolo11n_nms.onnx.det.json";
    if (!fs::exists(base_file)) return;

    UltralyticsDet model(modelfile.string(), cpu_option());
    if (!model.is_initialized()) return;

    auto img = ImageData::imread(imgf.string());
    if (img.empty()) return;

    std::vector<DetectionResult> results;
    REQUIRE(model.predict(img, &results, nullptr));
    require_no_diff(compare_detection(load_json(base_file)["results"], results));
}

TEST_CASE("Regression: yolo11n-seg segmentation", "[regression]") {
    auto modelfile = model_path("yolo11n-seg.onnx");
    if (!fs::exists(modelfile)) return;
    auto imgf = image_path("test_person.jpg");
    if (!fs::exists(imgf)) return;
    auto base_file = baseline_dir() / "yolo11n-seg.onnx.seg.json";
    if (!fs::exists(base_file)) return;

    UltralyticsSeg model(modelfile.string(), cpu_option());
    if (!model.is_initialized()) return;

    auto img = ImageData::imread(imgf.string());
    if (img.empty()) return;

    std::vector<InstanceSegResult> results;
    REQUIRE(model.predict(img, &results, nullptr));
    require_no_diff(compare_seg(load_json(base_file)["results"], results));
}

TEST_CASE("Regression: yolo11n-pose pose estimation", "[regression]") {
    auto modelfile = model_path("yolo11n-pose.onnx");
    if (!fs::exists(modelfile)) return;
    auto imgf = image_path("test_person.jpg");
    if (!fs::exists(imgf)) return;
    auto base_file = baseline_dir() / "yolo11n-pose.onnx.pose.json";
    if (!fs::exists(base_file)) return;

    UltralyticsPose model(modelfile.string(), cpu_option());
    if (!model.is_initialized()) return;

    auto img = ImageData::imread(imgf.string());
    if (img.empty()) return;

    std::vector<KeyPointsResult> results;
    REQUIRE(model.predict(img, &results, nullptr));
    require_no_diff(compare_pose(load_json(base_file)["results"], results));
}

TEST_CASE("Regression: yolo11n-obb obb detection", "[regression]") {
    auto modelfile = model_path("yolo11n-obb.onnx");
    if (!fs::exists(modelfile)) return;
    auto imgf = image_path("test_obb1.jpg");
    if (!fs::exists(imgf)) return;
    auto base_file = baseline_dir() / "yolo11n-obb.onnx.obb.json";
    if (!fs::exists(base_file)) return;

    UltralyticsObb model(modelfile.string(), cpu_option());
    if (!model.is_initialized()) return;

    auto img = ImageData::imread(imgf.string());
    if (img.empty()) return;

    std::vector<ObbResult> results;
    REQUIRE(model.predict(img, &results, nullptr));
    require_no_diff(compare_obb(load_json(base_file)["results"], results));
}

TEST_CASE("Regression: yolo11n-obb_nms obb detection", "[regression]") {
    auto modelfile = model_path("yolo11n-obb_nms.onnx");
    if (!fs::exists(modelfile)) return;
    auto imgf = image_path("test_obb1.jpg");
    if (!fs::exists(imgf)) return;
    auto base_file = baseline_dir() / "yolo11n-obb_nms.onnx.obb.json";
    if (!fs::exists(base_file)) return;

    UltralyticsObb model(modelfile.string(), cpu_option());
    if (!model.is_initialized()) return;

    auto img = ImageData::imread(imgf.string());
    if (img.empty()) return;

    std::vector<ObbResult> results;
    REQUIRE(model.predict(img, &results, nullptr));
    require_no_diff(compare_obb(load_json(base_file)["results"], results));
}

TEST_CASE("Regression: yolo11n-cls classification", "[regression]") {
    auto modelfile = model_path("yolo11n-cls.onnx");
    if (!fs::exists(modelfile)) return;
    auto imgf = image_path("test_person.jpg");
    if (!fs::exists(imgf)) return;
    auto base_file = baseline_dir() / "yolo11n-cls.onnx.cls.json";
    if (!fs::exists(base_file)) return;

    Classification model(modelfile.string(), cpu_option());
    if (!model.is_initialized()) return;

    auto img = ImageData::imread(imgf.string());
    if (img.empty()) return;

    ClassifyResult result;
    REQUIRE(model.predict(img, &result));
    require_no_diff(compare_cls(load_json(base_file)["results"], result));
}

TEST_CASE("Regression: scrfd face detection", "[regression]") {
    auto modelfile = model_path("face/scrfd_2.5g_bnkps_shape640x640.onnx");
    if (!fs::exists(modelfile)) return;
    auto imgf = image_path("test_face_detection.jpg");
    if (!fs::exists(imgf)) return;
    auto base_file = baseline_dir() / "scrfd_2.5g_bnkps_shape640x640.onnx.face_det.json";
    if (!fs::exists(base_file)) return;

    Scrfd model(modelfile.string(), cpu_option());
    if (!model.is_initialized()) return;

    auto img = ImageData::imread(imgf.string());
    if (img.empty()) return;

    std::vector<KeyPointsResult> results;
    REQUIRE(model.predict(img, &results, nullptr));
    require_no_diff(compare_pose(load_json(base_file)["results"], results));
}

TEST_CASE("Regression: ppocrv4 det + pre/raw", "[regression]") {
    auto modelfile = model_path("ocr/ppocrv4_mobile/det_infer.onnx");
    if (!fs::exists(modelfile)) return;
    auto imgf = image_path("test_ocr.png");
    if (!fs::exists(imgf)) return;
    auto base_file = baseline_dir() / "det_infer.onnx.ocr_det.json";
    if (!fs::exists(base_file)) return;

    DBDetector model(modelfile.string(), cpu_option());
    if (!model.is_initialized()) return;

    auto img = ImageData::imread(imgf.string());
    if (img.empty()) return;

    std::vector<std::array<int, 8>> boxes;
    REQUIRE(model.predict(img, &boxes, nullptr));
    require_no_diff(compare_ocr_det(load_json(base_file)["results"], boxes));

    auto& preproc = model.get_preprocessor();
    std::vector<Tensor> inputs;
    REQUIRE(preproc.apply({img}, &inputs));
    std::vector<Tensor> outputs;
    REQUIRE(model.infer(inputs, &outputs));
    REQUIRE_FALSE(inputs.empty());
    REQUIRE_FALSE(outputs.empty());
    check_tensors({baseline_dir() / "det_infer.onnx.pre.json",
                   baseline_dir() / "det_infer.onnx.raw.json"},
                  {inputs[0], outputs[0]});
}

TEST_CASE("Regression: ppocrv4 rec", "[regression]") {
    auto modelfile = model_path("ocr/ppocrv4_mobile/rec_infer.onnx");
    if (!fs::exists(modelfile)) return;
    // rec 模型输入为单行文本裁剪图，整图（test_ocr.png）会因输出空容器崩溃/失败
    auto imgf = image_path("test_ocr_recognition.jpg");
    if (!fs::exists(imgf)) return;
    auto base_file = baseline_dir() / "rec_infer.onnx.ocr_rec.json";
    if (!fs::exists(base_file)) return;
    auto dict = get_test_data() / "ppocrv4_dict.txt";
    if (!fs::exists(dict)) return;

    Recognizer model(modelfile.string(), dict.string(), cpu_option());
    if (!model.is_initialized()) return;

    auto img = ImageData::imread(imgf.string());
    if (img.empty()) return;

    std::string text;
    float score = 0;
    REQUIRE(model.predict(img, &text, &score, nullptr));
    require_no_diff(compare_ocr_rec(load_json(base_file)["results"], text, score));
}

TEST_CASE("Regression: ppocrv4 cls", "[regression]") {
    auto modelfile = model_path("ocr/ppocrv4_mobile/cls_infer.onnx");
    if (!fs::exists(modelfile)) return;
    auto imgf = image_path("test_ocr.png");
    if (!fs::exists(imgf)) return;
    auto base_file = baseline_dir() / "cls_infer.onnx.ocr_cls.json";
    if (!fs::exists(base_file)) return;

    Classifier model(modelfile.string(), cpu_option());
    if (!model.is_initialized()) return;

    auto img = ImageData::imread(imgf.string());
    if (img.empty()) return;

    int32_t label = -1;
    float score = 0;
    REQUIRE(model.predict(img, &label, &score));
    require_no_diff(compare_ocr_cls(load_json(base_file)["results"], label, score));
}
