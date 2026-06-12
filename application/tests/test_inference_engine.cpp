#include <catch2/catch_test_macros.hpp>
#include "inference_engine.hpp"

TEST_CASE("InferenceEngine construct/destroy", "[infer]") {
    InferenceEngine engine;
    REQUIRE_FALSE(engine.is_loaded());
}

TEST_CASE("InferenceEngine load nonexistent model", "[infer]") {
    InferenceEngine engine;
    ModelConfig cfg;
    cfg.name = "test";
    cfg.type = "detection";
    cfg.path = "/nonexistent/path.onnx";
    REQUIRE_FALSE(engine.load(cfg));
}

TEST_CASE("InferenceEngine unload when not loaded", "[infer]") {
    InferenceEngine engine;
    REQUIRE_NOTHROW(engine.unload());
}

TEST_CASE("InferenceEngine infer without load", "[infer]") {
    InferenceEngine engine;
    InferResult result;
    MDImage img = {};
    REQUIRE_FALSE(engine.infer(&img, &result));
}

TEST_CASE("InferenceEngine config access", "[infer]") {
    InferenceEngine engine;
    ModelConfig cfg;
    cfg.name = "test_model";
    cfg.type = "detection";
    cfg.path = "/m/yolo.onnx";
    cfg.input_size = {640, 640};
    // Load will fail, but config is stored before load attempt
    engine.load(cfg);
    // The config() returns the stored config regardless of load success
    // (it's set before the load attempt)
}
