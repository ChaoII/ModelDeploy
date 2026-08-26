#include <catch2/catch_test_macros.hpp>
#include "infer_group.hpp"
#include "csrc/vision/common/image_data.h"
#include <fstream>
#include <vector>

using modeldeploy::vision::ImageData;
using modeldeploy::vision::DetectionResult;

static ImageData tiny_image() {
    std::vector<uint8_t> buf(16 * 16 * 3, 0);
    return ImageData::from_raw(buf.data(), 16, 16, MdImageType::PKG_BGR_U8, true);
}

TEST_CASE("InferGroup empty by default", "[infer_group]") {
    InferGroup g;
    REQUIRE(g.empty());
}

TEST_CASE("InferGroup run_models on empty returns false", "[infer_group]") {
    InferGroup g;
    auto img = tiny_image();
    std::vector<std::pair<std::string, std::vector<DetectionResult>>> dets;
    REQUIRE_FALSE(g.run_models(img, &dets));
    REQUIRE(dets.empty());
}

TEST_CASE("InferGroup remove/clear on empty is safe", "[infer_group]") {
    InferGroup g;
    REQUIRE_FALSE(g.remove_model("nonexistent"));
    g.clear();
    REQUIRE(g.empty());
}

TEST_CASE("InferGroup add nonexistent model fails", "[infer_group]") {
    InferGroup g;
    ModelConfig m;
    m.name = "det";
    m.type = "detection";
    m.path = "/nonexistent.onnx";
    REQUIRE_FALSE(g.add_model(m, nullptr));
    REQUIRE(g.empty());
}

TEST_CASE("InferGroup det_model on empty returns nullptr", "[infer_group]") {
    InferGroup g;
    REQUIRE(g.det_model("det") == nullptr);
}

TEST_CASE("InferGroup load_models empty returns true", "[infer_group]") {
    InferGroup g;
    REQUIRE(g.load_models({}, nullptr));
    REQUIRE(g.empty());
}

// 集成：需要真实检测模型文件，无 test_data 时跳过
TEST_CASE("InferGroup real detection model integration", "[infer_group][integration]") {
    const std::string model = "E:/CLionProjects/ModelDeploy/test_data/test_models/yolo11n_nms.onnx";
    if (!std::ifstream(model).good()) {
        SKIP("real detection model absent; skipping integration");
    }
    InferGroup g;
    ModelConfig m;
    m.name = "yolo11n";
    m.type = "detection";
    m.path = model;
    m.device = "cpu";
    m.input_size = {640, 640};
    REQUIRE(g.add_model(m, nullptr));
    REQUIRE_FALSE(g.empty());
    REQUIRE(g.det_model("yolo11n") != nullptr);
}
