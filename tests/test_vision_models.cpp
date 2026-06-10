#include <catch2/catch_test_macros.hpp>
#include <filesystem>
#include <vector>
#include <string>
#include <array>
#include "csrc/vision.h"

namespace fs = std::filesystem;
using namespace modeldeploy::vision;
using namespace modeldeploy::vision::detection;
using namespace modeldeploy::vision::classification;
using namespace modeldeploy::vision::face;
using namespace modeldeploy::vision::ocr;

static fs::path get_test_data() {
    const char* env = std::getenv("TEST_DATA_DIR");
    if (env && *env) return fs::path(env) / "test_data";
    return fs::current_path() / "test_data";
}

static fs::path model_path(const std::string& rel) {
    return get_test_data() / "test_models" / rel;
}

static fs::path image_path(const std::string& name) {
    return get_test_data() / "test_images" / name;
}

static ImageData load_image(const std::string& name) {
    return ImageData::imread(image_path(name).string());
}

template<typename Model, typename Result>
static void test_model_predict(Model& model, const ImageData& img, std::vector<Result>* results) {
    REQUIRE(model.predict(img, results));
    REQUIRE_FALSE(results->empty());
}

// ==================== Classification ====================
TEST_CASE("Classification model", "[vision_models]") {
    auto modelfile = model_path("yolo11n-cls.onnx");
    if (!fs::exists(modelfile)) return;

    modeldeploy::RuntimeOption opt;
    opt.use_cpu();

    Classification model(modelfile.string(), opt);
    REQUIRE(model.name() == "Classification");

    auto img = load_image("test_person.jpg");
    if (img.empty()) return;

    ClassifyResult result;
    REQUIRE(model.predict(img, &result));
    REQUIRE(result.label_ids.size() > 0);
    REQUIRE(result.scores.size() > 0);
    REQUIRE(result.label_ids[0] >= 0);
    REQUIRE(result.scores[0] > 0);

    auto& preproc = model.get_preprocessor();
    auto& postproc = model.get_postprocessor();
}

// ==================== Ultralytics Detection ====================
TEST_CASE("UltralyticsDet model", "[vision_models]") {
    auto modelfile = model_path("yolo11n.onnx");
    if (!fs::exists(modelfile)) return;

    modeldeploy::RuntimeOption opt;
    opt.use_cpu();

    UltralyticsDet model(modelfile.string(), opt);
    REQUIRE(model.name() == "UltralyticsDet");

    auto img = load_image("test_detection0.jpg");
    if (img.empty()) return;

    std::vector<DetectionResult> results;
    REQUIRE(model.predict(img, &results, nullptr));
    REQUIRE(results.size() > 0);
    for (auto& r : results) {
        REQUIRE(r.box.width > 0);
        REQUIRE(r.box.height >= 0);
        REQUIRE(r.label_id >= 0);
        REQUIRE(r.score > 0);
    }
}

// ==================== Ultralytics Segmentation ====================
TEST_CASE("UltralyticsSeg model", "[vision_models]") {
    auto modelfile = model_path("yolo11n-seg.onnx");
    if (!fs::exists(modelfile)) return;

    modeldeploy::RuntimeOption opt;
    opt.use_cpu();

    UltralyticsSeg model(modelfile.string(), opt);
    REQUIRE(model.name() == "UltralyticsSeg");

    auto img = load_image("test_person.jpg");
    if (img.empty()) return;

    std::vector<InstanceSegResult> results;
    REQUIRE(model.predict(img, &results, nullptr));
    REQUIRE(results.size() > 0);
    for (auto& r : results) {
        REQUIRE(r.box.width > 0);
        REQUIRE(r.label_id >= 0);
        REQUIRE(r.score > 0);
    }
}

// ==================== Ultralytics Pose ====================
TEST_CASE("UltralyticsPose model", "[vision_models]") {
    auto modelfile = model_path("yolo11n-pose.onnx");
    if (!fs::exists(modelfile)) return;

    modeldeploy::RuntimeOption opt;
    opt.use_cpu();

    UltralyticsPose model(modelfile.string(), opt);
    REQUIRE(model.name() == "UltralyticsPose");

    auto img = load_image("test_person.jpg");
    if (img.empty()) return;

    std::vector<KeyPointsResult> results;
    REQUIRE(model.predict(img, &results, nullptr));
    REQUIRE(results.size() > 0);
    for (auto& r : results) {
        REQUIRE(r.box.width > 0);
        REQUIRE(r.keypoints.size() > 0);
    }
}

// ==================== Ultralytics OBB ====================
TEST_CASE("UltralyticsObb model", "[vision_models]") {
    auto modelfile = model_path("yolo11n-obb.onnx");
    if (!fs::exists(modelfile)) return;

    modeldeploy::RuntimeOption opt;
    opt.use_cpu();

    UltralyticsObb model(modelfile.string(), opt);
    REQUIRE(model.name() == "UltralyticsObb");

    auto img = load_image("test_obb.jpg");
    if (img.empty()) {
        img = load_image("test_detection0.jpg");
    }
    if (img.empty()) return;

    std::vector<ObbResult> results;
    REQUIRE(model.predict(img, &results, nullptr));
    REQUIRE(results.size() > 0);
    for (auto& r : results) {
        REQUIRE(r.rotated_box.xc > 0);
        REQUIRE(r.label_id >= 0);
        REQUIRE(r.score > 0);
    }
}

// ==================== Batch Predict ====================
TEST_CASE("Batch predict for vision models", "[vision_models]") {
    auto modelfile = model_path("yolo11n.onnx");
    if (!fs::exists(modelfile)) return;

    modeldeploy::RuntimeOption opt;
    opt.use_cpu();

    UltralyticsDet model(modelfile.string(), opt);

    auto img1 = load_image("test_detection0.jpg");
    auto img2 = load_image("test_person.jpg");
    if (img1.empty() || img2.empty()) return;

    std::vector<std::vector<DetectionResult>> results;
    REQUIRE(model.batch_predict({img1, img2}, &results, nullptr));
    REQUIRE(results.size() == 2);
    REQUIRE(results[0].size() > 0);
    REQUIRE(results[1].size() > 0);
}

// ==================== Face Models ====================
TEST_CASE("Scrfd face detection model", "[vision_models]") {
    auto modelfile = model_path("face/scrfd_2.5g_bnkps_shape640x640.onnx");
    if (!fs::exists(modelfile)) return;

    modeldeploy::RuntimeOption opt;
    opt.use_cpu();

    Scrfd model(modelfile.string(), opt);

    auto img = load_image("test_face_detection.jpg");
    if (img.empty()) return;

    std::vector<KeyPointsResult> results;
    REQUIRE(model.predict(img, &results, nullptr));
    REQUIRE(results.size() > 0);
}

TEST_CASE("SeetaFaceAge model", "[vision_models]") {
    auto modelfile = model_path("face/age_predictor.onnx");
    if (!fs::exists(modelfile)) return;

    modeldeploy::RuntimeOption opt;
    opt.use_cpu();

    SeetaFaceAge model(modelfile.string(), opt);

    auto img = load_image("test_face.jpg");
    if (img.empty()) return;

    int age = -1;
    REQUIRE(model.predict(img, &age));
    REQUIRE(age >= 0);
}

TEST_CASE("SeetaFaceGender model", "[vision_models]") {
    auto modelfile = model_path("face/gender_predictor.onnx");
    if (!fs::exists(modelfile)) return;

    modeldeploy::RuntimeOption opt;
    opt.use_cpu();

    SeetaFaceGender model(modelfile.string(), opt);

    auto img = load_image("test_face_gender.jpg");
    if (img.empty()) return;

    int gender = -1;
    REQUIRE(model.predict(img, &gender));
    REQUIRE(gender >= 0);
}

// ==================== OCR Models ====================
TEST_CASE("OCR DBDetector model", "[vision_models]") {
    auto modelfile = model_path("ocr") / "ppocrv4_mobile" / "det" / "inference.onnx";
    if (!fs::exists(modelfile)) {
        modelfile = model_path("ocr") / "ppocrv5_mobile" / "det" / "inference.onnx";
    }
    if (!fs::exists(modelfile)) return;

    modeldeploy::RuntimeOption opt;
    opt.use_cpu();

    DBDetector model(modelfile.string(), opt);
    REQUIRE(model.name() == "DBDetector");

    auto img = load_image("test_ocr.png");
    if (img.empty()) return;

    std::vector<std::array<int, 8>> boxes;
    REQUIRE(model.predict(img, &boxes, nullptr));
    REQUIRE(boxes.size() > 0);
}

TEST_CASE("OCR Classifier model", "[vision_models]") {
    auto modelfile = model_path("ocr") / "ppocrv4_mobile" / "cls" / "inference.onnx";
    if (!fs::exists(modelfile)) {
        modelfile = model_path("ocr") / "ppocrv5_mobile" / "cls" / "inference.onnx";
    }
    if (!fs::exists(modelfile)) return;

    modeldeploy::RuntimeOption opt;
    opt.use_cpu();

    Classifier model(modelfile.string(), opt);

    auto img = load_image("test_ocr.png");
    if (img.empty()) return;

    int32_t cls_label = -1;
    float cls_score = 0;
    REQUIRE(model.predict(img, &cls_label, &cls_score));
    REQUIRE(cls_label >= 0);
}

TEST_CASE("OCR Recognizer model", "[vision_models]") {
    auto modelfile = model_path("ocr") / "ppocrv4_mobile" / "rec" / "inference.onnx";
    if (!fs::exists(modelfile)) {
        modelfile = model_path("ocr") / "ppocrv5_mobile" / "rec" / "inference.onnx";
    }
    if (!fs::exists(modelfile)) return;

    auto dict = get_test_data() / "ppocrv4_dict.txt";
    if (!fs::exists(dict)) return;

    modeldeploy::RuntimeOption opt;
    opt.use_cpu();

    Recognizer model(modelfile.string(), dict.string(), opt);

    auto img = load_image("test_ocr.png");
    if (img.empty()) return;

    std::string text;
    float score = 0;
    REQUIRE(model.predict(img, &text, &score, nullptr));
    REQUIRE_FALSE(text.empty());
    REQUIRE(score > 0);
}

// ==================== Preprocessor access ====================
TEST_CASE("Preprocessor/Postprocessor access", "[vision_models]") {
    auto modelfile = model_path("yolo11n.onnx");
    if (!fs::exists(modelfile)) return;

    modeldeploy::RuntimeOption opt;
    opt.use_cpu();

    UltralyticsDet model(modelfile.string(), opt);
    auto& preproc = model.get_preprocessor();
    auto& postproc = model.get_postprocessor();

    preproc.set_size({640, 640});
    auto size = preproc.get_size();
    REQUIRE(size.size() == 2);
}
