#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>
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
    auto modelfile = model_path("onnx/yolo11n/yolo11n-cls.onnx");
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
    auto modelfile = model_path("onnx/yolo11n/yolo11n.onnx");
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

// 同一 NV12 缓冲：predict(ImageData) 的 NV12 分叉是统一单入口，须能零拷贝预处并产出合理结果
// [model]：需模型文件，缺文件时跳过（不自红）
TEST_CASE("UltralyticsDet predict(ImageData) on NV12 device frame", "[model]") {
    auto modelfile = model_path("onnx/yolo11n/yolo11n.onnx");
    if (!fs::exists(modelfile)) return;

    modeldeploy::RuntimeOption opt;
    opt.use_cpu();
    UltralyticsDet model(modelfile.string(), opt);

    auto img = load_image("test_detection0.jpg");
    if (img.empty()) return;
    const int w = img.width(), h = img.height();
    const int step_src = img.plane(0).step > 0 ? img.plane(0).step : w * 3;

    // BGR → NV12：Y 取 luma（保留场景结构），UV 置中性灰 128
    std::vector<uint8_t> y(static_cast<size_t>(w) * h);
    for (int i = 0; i < h; ++i) {
        const uint8_t* row = img.plane(0).data + static_cast<size_t>(i) * step_src;
        for (int j = 0; j < w; ++j) {
            const uint8_t b = row[j * 3 + 0], g = row[j * 3 + 1], r = row[j * 3 + 2];
            y[static_cast<size_t>(i) * w + j] =
                static_cast<uint8_t>((77 * r + 150 * g + 29 * b + 128) >> 8);
        }
    }
    std::vector<uint8_t> uv(static_cast<size_t>(w) * h / 2, 128);

    ImageData::Plane pl[2] = {{y.data(), w}, {uv.data(), w}};
    ImageData frame = ImageData::from_planes(pl, 2, MdImageType::NV12, w, h, modeldeploy::Device::CPU);
    REQUIRE(frame.format() == MdImageType::NV12);
    REQUIRE(frame.plane_count() == 2);

    std::vector<DetectionResult> r_predict;
    REQUIRE(model.predict(frame, &r_predict, nullptr));
    REQUIRE(r_predict.size() > 0);
    for (auto& r : r_predict) {
        REQUIRE(r.box.width > 0);
        REQUIRE(r.box.height >= 0);
        REQUIRE(r.label_id >= 0);
        REQUIRE(r.score > 0);
    }
}

// ==================== Ultralytics Segmentation ====================
TEST_CASE("UltralyticsSeg model", "[vision_models]") {
    auto modelfile = model_path("onnx/yolo11n/yolo11n-seg.onnx");
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
    auto modelfile = model_path("onnx/yolo11n/yolo11n-pose.onnx");
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
    auto modelfile = model_path("onnx/yolo11n/yolo11n-obb.onnx");
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
    auto modelfile = model_path("onnx/yolo11n/yolo11n.onnx");
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
    auto modelfile = model_path("onnx/face/scrfd_2.5g_bnkps_shape640x640.onnx");
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
    auto modelfile = model_path("onnx/face/age_predictor.onnx");
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
    auto modelfile = model_path("onnx/face/gender_predictor.onnx");
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
    auto modelfile = model_path("onnx/ocr/ppocrv4_mobile/det_infer.onnx");
    if (!fs::exists(modelfile)) {
        modelfile = model_path("onnx/ocr/ppocrv5_mobile/det_infer.onnx");
    }
    if (!fs::exists(modelfile)) return;

    modeldeploy::RuntimeOption opt;
    opt.use_cpu();

    DBDetector model(modelfile.string(), opt);
    REQUIRE(model.name() == "ppocr/ocr_det");

    auto img = load_image("test_ocr.png");
    if (img.empty()) return;

    std::vector<std::array<int, 8>> boxes;
    REQUIRE(model.predict(img, &boxes, nullptr));
    REQUIRE(boxes.size() > 0);
}

TEST_CASE("OCR Classifier model", "[vision_models]") {
    auto modelfile = model_path("onnx/ocr/ppocrv4_mobile/cls_infer.onnx");
    if (!fs::exists(modelfile)) {
        modelfile = model_path("onnx/ocr/ppocrv5_mobile/cls_infer.onnx");
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
    auto modelfile = model_path("onnx/ocr/ppocrv4_mobile/rec_infer.onnx");
    if (!fs::exists(modelfile)) {
        modelfile = model_path("onnx/ocr/ppocrv5_mobile/rec_infer.onnx");
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
    auto modelfile = model_path("onnx/yolo11n/yolo11n.onnx");
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
