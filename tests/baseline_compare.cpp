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

static fs::path baseline_root() { return get_test_data().parent_path() / "tests" / "baselines"; }

static fs::path baseline_dir(const std::string& backend) {
    return baseline_root() / backend;
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
    opt.use_trt_backend();
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
    auto base_file = baseline_dir("ort") / "yolo11n.onnx.det.json";
    if (!fs::exists(base_file)) return;

    UltralyticsDet model(modelfile.string(), cpu_option());
    REQUIRE(model.is_initialized());

    auto img = ImageData::imread(imgf.string());
    REQUIRE_FALSE(img.empty());

    std::vector<DetectionResult> results;
    REQUIRE(model.predict(img, &results, nullptr));
    require_no_diff(compare_detection(load_json(base_file)["results"], results));

    compare_yolo_pre_raw(model, img,
                         baseline_dir("ort") / "yolo11n.onnx.pre.json",
                         baseline_dir("ort") / "yolo11n.onnx.raw.json");
}

TEST_CASE("Regression: yolo11n_nms detection", "[regression]") {
    auto modelfile = model_path("yolo11n_nms.onnx");
    if (!fs::exists(modelfile)) return;
    auto imgf = image_path("test_detection0.jpg");
    if (!fs::exists(imgf)) return;
    auto base_file = baseline_dir("ort") / "yolo11n_nms.onnx.det.json";
    if (!fs::exists(base_file)) return;

    UltralyticsDet model(modelfile.string(), cpu_option());
    REQUIRE(model.is_initialized());

    auto img = ImageData::imread(imgf.string());
    REQUIRE_FALSE(img.empty());

    std::vector<DetectionResult> results;
    REQUIRE(model.predict(img, &results, nullptr));
    require_no_diff(compare_detection(load_json(base_file)["results"], results));
}

TEST_CASE("Regression: yolo11n detection MNN", "[regression][backend:mnn]") {
    auto modelfile = model_path("yolo11n.mnn", "mnn");
    if (!fs::exists(modelfile)) return;
    auto imgf = image_path("test_detection0.jpg");
    if (!fs::exists(imgf)) return;
    auto ort_file = baseline_dir("ort") / "yolo11n.onnx.det.json";
    auto self_file = baseline_dir("mnn") / "yolo11n.mnn.det.json";
    if (!fs::exists(ort_file)) return;

    UltralyticsDet model(modelfile.string(), cpu_option());
    REQUIRE(model.is_initialized());

    auto img = ImageData::imread(imgf.string());
    REQUIRE_FALSE(img.empty());

    std::vector<DetectionResult> results;
    REQUIRE(model.predict(img, &results, nullptr));

    if (fs::exists(self_file)) {
        require_no_diff(compare_detection(load_json(self_file)["results"], results));
    }
    warn_diff(compare_detection(load_json(ort_file)["results"], results), "yolo11n.mnn vs ORT baseline");
}

TEST_CASE("Regression: yolo11n_nms detection MNN", "[regression][backend:mnn]") {
    auto modelfile = model_path("yolo11n_nms.mnn", "mnn");
    if (!fs::exists(modelfile)) return;
    auto imgf = image_path("test_detection0.jpg");
    if (!fs::exists(imgf)) return;
    auto ort_file = baseline_dir("ort") / "yolo11n_nms.onnx.det.json";
    auto self_file = baseline_dir("mnn") / "yolo11n_nms.mnn.det.json";
    if (!fs::exists(ort_file)) return;

    UltralyticsDet model(modelfile.string(), cpu_option());
    REQUIRE(model.is_initialized());

    auto img = ImageData::imread(imgf.string());
    REQUIRE_FALSE(img.empty());

    std::vector<DetectionResult> results;
    REQUIRE(model.predict(img, &results, nullptr));

    if (fs::exists(self_file)) {
        require_no_diff(compare_detection(load_json(self_file)["results"], results));
    }
    warn_diff(compare_detection(load_json(ort_file)["results"], results), "yolo11n_nms.mnn vs ORT baseline");
}

TEST_CASE("Regression: yolo11n detection TRT", "[regression][backend:trt]") {
    auto modelfile = model_path("yolo11n.engine", "trt");
    if (!fs::exists(modelfile)) return;
    auto imgf = image_path("test_detection0.jpg");
    if (!fs::exists(imgf)) return;
    auto ort_file = baseline_dir("ort") / "yolo11n.onnx.det.json";
    auto self_file = baseline_dir("trt") / "yolo11n.engine.det.json";
    if (!fs::exists(ort_file)) return;

    UltralyticsDet model(modelfile.string(), trt_option());
    REQUIRE(model.is_initialized());

    auto img = ImageData::imread(imgf.string());
    REQUIRE_FALSE(img.empty());

    std::vector<DetectionResult> results;
    REQUIRE(model.predict(img, &results, nullptr));

    if (fs::exists(self_file)) {
        require_no_diff(compare_detection(load_json(self_file)["results"], results));
    }
    warn_diff(compare_detection(load_json(ort_file)["results"], results), "yolo11n.engine vs ORT baseline");
}

TEST_CASE("Regression: yolo11n_nms detection TRT", "[regression][backend:trt]") {
    auto modelfile = model_path("yolo11n_nms.engine", "trt");
    if (!fs::exists(modelfile)) return;
    auto imgf = image_path("test_detection0.jpg");
    if (!fs::exists(imgf)) return;
    auto ort_file = baseline_dir("ort") / "yolo11n_nms.onnx.det.json";
    auto self_file = baseline_dir("trt") / "yolo11n_nms.engine.det.json";
    if (!fs::exists(ort_file)) return;

    UltralyticsDet model(modelfile.string(), trt_option());
    REQUIRE(model.is_initialized());

    auto img = ImageData::imread(imgf.string());
    REQUIRE_FALSE(img.empty());

    std::vector<DetectionResult> results;
    REQUIRE(model.predict(img, &results, nullptr));

    if (fs::exists(self_file)) {
        require_no_diff(compare_detection(load_json(self_file)["results"], results));
    }
    warn_diff(compare_detection(load_json(ort_file)["results"], results), "yolo11n_nms.engine vs ORT baseline");
}

TEST_CASE("Regression: yolo11n-seg_nms segmentation TRT", "[regression][backend:trt]") {
    auto modelfile = model_path("yolo11n-seg_nms.engine", "trt");
    if (!fs::exists(modelfile)) return;
    auto imgf = image_path("test_person.jpg");
    if (!fs::exists(imgf)) return;
    auto ort_file = baseline_dir("ort") / "yolo11n-seg_nms.onnx.seg.json";
    auto self_file = baseline_dir("trt") / "yolo11n-seg_nms.engine.seg.json";
    if (!fs::exists(ort_file)) return;

    UltralyticsSeg model(modelfile.string(), trt_option());
    REQUIRE(model.is_initialized());

    auto img = ImageData::imread(imgf.string());
    REQUIRE_FALSE(img.empty());

    std::vector<InstanceSegResult> results;
    REQUIRE(model.predict(img, &results, nullptr));

    if (fs::exists(self_file)) {
        require_no_diff(compare_seg(load_json(self_file)["results"], results));
    }
    warn_diff(compare_seg(load_json(ort_file)["results"], results), "yolo11n-seg_nms.engine vs ORT baseline");
}

TEST_CASE("Regression: yolo11n-cls classification TRT", "[regression][backend:trt]") {
    auto modelfile = model_path("yolo11n-cls.engine", "trt");
    if (!fs::exists(modelfile)) return;
    auto imgf = image_path("test_person.jpg");
    if (!fs::exists(imgf)) return;
    auto ort_file = baseline_dir("ort") / "yolo11n-cls.onnx.cls.json";
    auto self_file = baseline_dir("trt") / "yolo11n-cls.engine.cls.json";
    if (!fs::exists(ort_file)) return;

    Classification model(modelfile.string(), trt_option());
    REQUIRE(model.is_initialized());

    auto img = ImageData::imread(imgf.string());
    REQUIRE_FALSE(img.empty());

    ClassifyResult result;
    REQUIRE(model.predict(img, &result));

    if (fs::exists(self_file)) {
        require_no_diff(compare_cls(load_json(self_file)["results"], result));
    }
    warn_diff(compare_cls(load_json(ort_file)["results"], result), "yolo11n-cls.engine vs ORT baseline");
}

TEST_CASE("Regression: yolo11n-obb obb detection TRT", "[regression][backend:trt]") {
    auto modelfile = model_path("yolo11n-obb.engine", "trt");
    if (!fs::exists(modelfile)) return;
    auto imgf = image_path("test_obb1.jpg");
    if (!fs::exists(imgf)) return;
    auto ort_file = baseline_dir("ort") / "yolo11n-obb.onnx.obb.json";
    auto self_file = baseline_dir("trt") / "yolo11n-obb.engine.obb.json";
    if (!fs::exists(ort_file)) return;

    UltralyticsObb model(modelfile.string(), trt_option());
    REQUIRE(model.is_initialized());

    auto img = ImageData::imread(imgf.string());
    REQUIRE_FALSE(img.empty());

    std::vector<ObbResult> results;
    REQUIRE(model.predict(img, &results, nullptr));

    if (fs::exists(self_file)) {
        require_no_diff(compare_obb(load_json(self_file)["results"], results));
    }
    warn_diff(compare_obb(load_json(ort_file)["results"], results), "yolo11n-obb.engine vs ORT baseline");
}

TEST_CASE("Regression: yolo11n-obb_nms obb detection TRT", "[regression][backend:trt]") {
    auto modelfile = model_path("yolo11n-obb_nms.engine", "trt");
    if (!fs::exists(modelfile)) return;
    auto imgf = image_path("test_obb1.jpg");
    if (!fs::exists(imgf)) return;
    auto ort_file = baseline_dir("ort") / "yolo11n-obb_nms.onnx.obb.json";
    auto self_file = baseline_dir("trt") / "yolo11n-obb_nms.engine.obb.json";
    if (!fs::exists(ort_file)) return;

    UltralyticsObb model(modelfile.string(), trt_option());
    REQUIRE(model.is_initialized());

    auto img = ImageData::imread(imgf.string());
    REQUIRE_FALSE(img.empty());

    std::vector<ObbResult> results;
    REQUIRE(model.predict(img, &results, nullptr));

    if (fs::exists(self_file)) {
        require_no_diff(compare_obb(load_json(self_file)["results"], results));
    }
    warn_diff(compare_obb(load_json(ort_file)["results"], results), "yolo11n-obb_nms.engine vs ORT baseline");
}

TEST_CASE("Regression: yolo11n-pose pose estimation TRT", "[regression][backend:trt]") {
    auto modelfile = model_path("yolo11n-pose.engine", "trt");
    if (!fs::exists(modelfile)) return;
    auto imgf = image_path("test_person.jpg");
    if (!fs::exists(imgf)) return;
    auto ort_file = baseline_dir("ort") / "yolo11n-pose.onnx.pose.json";
    auto self_file = baseline_dir("trt") / "yolo11n-pose.engine.pose.json";
    if (!fs::exists(ort_file)) return;

    UltralyticsPose model(modelfile.string(), trt_option());
    REQUIRE(model.is_initialized());

    auto img = ImageData::imread(imgf.string());
    REQUIRE_FALSE(img.empty());

    std::vector<KeyPointsResult> results;
    REQUIRE(model.predict(img, &results, nullptr));

    if (fs::exists(self_file)) {
        require_no_diff(compare_pose(load_json(self_file)["results"], results));
    }
    warn_diff(compare_pose(load_json(ort_file)["results"], results), "yolo11n-pose.engine vs ORT baseline");
}

TEST_CASE("Regression: yolo11n-pose_nms pose estimation TRT", "[regression][backend:trt]") {
    auto modelfile = model_path("yolo11n-pose_nms.engine", "trt");
    if (!fs::exists(modelfile)) return;
    auto imgf = image_path("test_person.jpg");
    if (!fs::exists(imgf)) return;
    auto ort_file = baseline_dir("ort") / "yolo11n-pose_nms.onnx.pose.json";
    auto self_file = baseline_dir("trt") / "yolo11n-pose_nms.engine.pose.json";
    if (!fs::exists(ort_file)) return;

    UltralyticsPose model(modelfile.string(), trt_option());
    REQUIRE(model.is_initialized());

    auto img = ImageData::imread(imgf.string());
    REQUIRE_FALSE(img.empty());

    std::vector<KeyPointsResult> results;
    REQUIRE(model.predict(img, &results, nullptr));

    if (fs::exists(self_file)) {
        require_no_diff(compare_pose(load_json(self_file)["results"], results));
    }
    warn_diff(compare_pose(load_json(ort_file)["results"], results), "yolo11n-pose_nms.engine vs ORT baseline");
}

TEST_CASE("Regression: yolo11n-seg segmentation TRT", "[regression][backend:trt]") {
    auto modelfile = model_path("yolo11n-seg.engine", "trt");
    if (!fs::exists(modelfile)) return;
    auto imgf = image_path("test_person.jpg");
    if (!fs::exists(imgf)) return;
    auto ort_file = baseline_dir("ort") / "yolo11n-seg.onnx.seg.json";
    auto self_file = baseline_dir("trt") / "yolo11n-seg.engine.seg.json";
    if (!fs::exists(ort_file)) return;

    UltralyticsSeg model(modelfile.string(), trt_option());
    REQUIRE(model.is_initialized());

    auto img = ImageData::imread(imgf.string());
    REQUIRE_FALSE(img.empty());

    std::vector<InstanceSegResult> results;
    REQUIRE(model.predict(img, &results, nullptr));

    if (fs::exists(self_file)) {
        require_no_diff(compare_seg(load_json(self_file)["results"], results));
    }
    warn_diff(compare_seg(load_json(ort_file)["results"], results), "yolo11n-seg.engine vs ORT baseline");
}

TEST_CASE("Regression: yolo11n-seg segmentation", "[regression]") {
    auto modelfile = model_path("yolo11n-seg.onnx");
    if (!fs::exists(modelfile)) return;
    auto imgf = image_path("test_person.jpg");
    if (!fs::exists(imgf)) return;
    auto base_file = baseline_dir("ort") / "yolo11n-seg.onnx.seg.json";
    if (!fs::exists(base_file)) return;

    UltralyticsSeg model(modelfile.string(), cpu_option());
    REQUIRE(model.is_initialized());

    auto img = ImageData::imread(imgf.string());
    REQUIRE_FALSE(img.empty());

    std::vector<InstanceSegResult> results;
    REQUIRE(model.predict(img, &results, nullptr));
    require_no_diff(compare_seg(load_json(base_file)["results"], results));
}

TEST_CASE("Regression: yolo11n-pose pose estimation", "[regression]") {
    auto modelfile = model_path("yolo11n-pose.onnx");
    if (!fs::exists(modelfile)) return;
    auto imgf = image_path("test_person.jpg");
    if (!fs::exists(imgf)) return;
    auto base_file = baseline_dir("ort") / "yolo11n-pose.onnx.pose.json";
    if (!fs::exists(base_file)) return;

    UltralyticsPose model(modelfile.string(), cpu_option());
    REQUIRE(model.is_initialized());

    auto img = ImageData::imread(imgf.string());
    REQUIRE_FALSE(img.empty());

    std::vector<KeyPointsResult> results;
    REQUIRE(model.predict(img, &results, nullptr));
    require_no_diff(compare_pose(load_json(base_file)["results"], results));
}

TEST_CASE("Regression: yolo11n-obb obb detection", "[regression]") {
    auto modelfile = model_path("yolo11n-obb.onnx");
    if (!fs::exists(modelfile)) return;
    auto imgf = image_path("test_obb1.jpg");
    if (!fs::exists(imgf)) return;
    auto base_file = baseline_dir("ort") / "yolo11n-obb.onnx.obb.json";
    if (!fs::exists(base_file)) return;

    UltralyticsObb model(modelfile.string(), cpu_option());
    REQUIRE(model.is_initialized());

    auto img = ImageData::imread(imgf.string());
    REQUIRE_FALSE(img.empty());

    std::vector<ObbResult> results;
    REQUIRE(model.predict(img, &results, nullptr));
    require_no_diff(compare_obb(load_json(base_file)["results"], results));
}

TEST_CASE("Regression: yolo11n-obb_nms obb detection", "[regression]") {
    auto modelfile = model_path("yolo11n-obb_nms.onnx");
    if (!fs::exists(modelfile)) return;
    auto imgf = image_path("test_obb1.jpg");
    if (!fs::exists(imgf)) return;
    auto base_file = baseline_dir("ort") / "yolo11n-obb_nms.onnx.obb.json";
    if (!fs::exists(base_file)) return;

    UltralyticsObb model(modelfile.string(), cpu_option());
    REQUIRE(model.is_initialized());

    auto img = ImageData::imread(imgf.string());
    REQUIRE_FALSE(img.empty());

    std::vector<ObbResult> results;
    REQUIRE(model.predict(img, &results, nullptr));
    require_no_diff(compare_obb(load_json(base_file)["results"], results));
}

TEST_CASE("Regression: yolo11n-cls classification", "[regression]") {
    auto modelfile = model_path("yolo11n-cls.onnx");
    if (!fs::exists(modelfile)) return;
    auto imgf = image_path("test_person.jpg");
    if (!fs::exists(imgf)) return;
    auto base_file = baseline_dir("ort") / "yolo11n-cls.onnx.cls.json";
    if (!fs::exists(base_file)) return;

    Classification model(modelfile.string(), cpu_option());
    REQUIRE(model.is_initialized());

    auto img = ImageData::imread(imgf.string());
    REQUIRE_FALSE(img.empty());

    ClassifyResult result;
    REQUIRE(model.predict(img, &result));
    require_no_diff(compare_cls(load_json(base_file)["results"], result));
}

TEST_CASE("Regression: scrfd face detection", "[regression]") {
    auto modelfile = model_path("face/scrfd_2.5g_bnkps_shape640x640.onnx");
    if (!fs::exists(modelfile)) return;
    auto imgf = image_path("test_face_detection.jpg");
    if (!fs::exists(imgf)) return;
    auto base_file = baseline_dir("ort") / "scrfd_2.5g_bnkps_shape640x640.onnx.face_det.json";
    if (!fs::exists(base_file)) return;

    Scrfd model(modelfile.string(), cpu_option());
    REQUIRE(model.is_initialized());

    auto img = ImageData::imread(imgf.string());
    REQUIRE_FALSE(img.empty());

    std::vector<KeyPointsResult> results;
    REQUIRE(model.predict(img, &results, nullptr));
    require_no_diff(compare_pose(load_json(base_file)["results"], results));
}

TEST_CASE("Regression: ppocrv4 det + pre/raw", "[regression]") {
    auto modelfile = model_path("ocr/ppocrv4_mobile/det_infer.onnx");
    if (!fs::exists(modelfile)) return;
    auto imgf = image_path("test_ocr.png");
    if (!fs::exists(imgf)) return;
    auto base_file = baseline_dir("ort") / "det_infer.onnx.ocr_det.json";
    if (!fs::exists(base_file)) return;

    DBDetector model(modelfile.string(), cpu_option());
    REQUIRE(model.is_initialized());

    auto img = ImageData::imread(imgf.string());
    REQUIRE_FALSE(img.empty());

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
    check_tensors({baseline_dir("ort") / "det_infer.onnx.pre.json",
                   baseline_dir("ort") / "det_infer.onnx.raw.json"},
                  {inputs[0], outputs[0]});
}

TEST_CASE("Regression: ppocrv4 rec", "[regression]") {
    auto modelfile = model_path("ocr/ppocrv4_mobile/rec_infer.onnx");
    if (!fs::exists(modelfile)) return;
    // rec 模型输入为单行文本裁剪图，整图（test_ocr.png）会因输出空容器崩溃/失败
    auto imgf = image_path("test_ocr_recognition.jpg");
    if (!fs::exists(imgf)) return;
    auto base_file = baseline_dir("ort") / "rec_infer.onnx.ocr_rec.json";
    if (!fs::exists(base_file)) return;
    auto dict = get_test_data() / "ppocrv4_dict.txt";
    if (!fs::exists(dict)) return;

    Recognizer model(modelfile.string(), dict.string(), cpu_option());
    REQUIRE(model.is_initialized());

    auto img = ImageData::imread(imgf.string());
    REQUIRE_FALSE(img.empty());

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
    auto base_file = baseline_dir("ort") / "cls_infer.onnx.ocr_cls.json";
    if (!fs::exists(base_file)) return;

    Classifier model(modelfile.string(), cpu_option());
    REQUIRE(model.is_initialized());

    auto img = ImageData::imread(imgf.string());
    REQUIRE_FALSE(img.empty());

    int32_t label = -1;
    float score = 0;
    REQUIRE(model.predict(img, &label, &score));
    require_no_diff(compare_ocr_cls(load_json(base_file)["results"], label, score));
}
