#include <catch2/catch_test_macros.hpp>
#include <fstream>
#include "inference_engine.hpp"

TEST_CASE("InferenceEngine loads pedestrian_attribute pipeline", "[agent][pipeline]") {
    if (!std::ifstream("test_data/test_models/onnx/zhgd_det.onnx").good() ||
        !std::ifstream("test_data/test_models/onnx/zhgd_ml.onnx").good())
        SKIP("test data absent");
    ModelConfig cfg;
    cfg.name = "pa";
    cfg.type = "pedestrian_attribute";
    cfg.backend = "ort";
    cfg.device = "cpu";
    cfg.path = "test_data/test_models/onnx/zhgd_det.onnx";
    cfg.rec_path = "test_data/test_models/onnx/zhgd_ml.onnx";
    cfg.input_size = {1280, 1280};
    cfg.confidence_threshold = 0.5f;
    cfg.attributes = {"safety_helmet", "reflective_vest", "safety_rope", "work_uniform"};

    InferenceEngine eng;
    REQUIRE(eng.load(cfg));
    auto img = modeldeploy::vision::ImageData::imread("test_data/test_images/test_pedestrian_attribute1.jpg");
    REQUIRE(!img.empty());
    InferResult r;
    REQUIRE(eng.infer(img, &r));
    REQUIRE(r.type == "pedestrian_attribute");
    REQUIRE(!r.attributes.empty());
    REQUIRE(!r.attributes[0].attr_scores.empty());
}
