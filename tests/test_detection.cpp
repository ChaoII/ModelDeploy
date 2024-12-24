//
// Created by AC on 2024-12-23.
//


#include "../src/detection_capi.h"
#include "../src/utils.h"
#include <catch2/catch_test_macros.hpp>
#include <filesystem>

StatusCode test_create_detection_model(const char *model_path, ModelFormat format) {
    WModel model;
    return create_detection_model(&model, model_path, 8, format);
}

StatusCode test_set_detection_input_size(const char *model_path, ModelFormat format) {
    WModel model;
    create_detection_model(&model, model_path, 8, format);
    return set_detection_input_size(&model, {1440, 1440});
}


StatusCode test_detection_model_predict() {
    WModel model;
    create_detection_model(&model, "best.onnx", 8);
    set_detection_input_size(&model, {1440, 1440});
    auto im = read_image("test_detection.png");
    WDetectionResults result;
    return detection_predict(&model, &result, im, 1, {255, 255, 0}, 0.5, 1);
}

void test_release_detection_result1() {
    WDetectionResults result{nullptr, 0};
    free_detection_result(&result);
}


void test_release_detection_result2() {
    WDetectionResults result;
    result.size = 2;
    result.data = (WDetectionResult *) malloc(sizeof(WDetectionResult) * result.size);
    result.data[0].box = {0, 0, 100, 100};
    result.data[1].box = {100, 100, 200, 200};
    result.data[0].label_id = 1;
    result.data[1].label_id = 2;
    result.data[0].score = 0.5;
    result.data[1].score = 0.6;
    free_detection_result(&result);
}

void test_release_detection_model() {
    WModel model;
    create_detection_model(&model, "best.onnx", 8);
    return free_detection_model(&model);
}

TEST_CASE("test create detection model function", "[create_detection_model]") {
    REQUIRE (test_create_detection_model("best.onnx", ModelFormat::ONNX) == StatusCode::Success);
    REQUIRE (test_create_detection_model("ppyoloe", ModelFormat::PaddlePaddle) == StatusCode::Success);
    REQUIRE (test_create_detection_model("picodet_lcnet", ModelFormat::PaddlePaddle) == StatusCode::Success);
}

TEST_CASE("test set detection input size function", "[set_detection_input_size]") {
    REQUIRE (test_set_detection_input_size("best.onnx", ModelFormat::ONNX) == StatusCode::Success);
    REQUIRE (test_set_detection_input_size("ppyoloe", ModelFormat::PaddlePaddle) == StatusCode::CallError);
}

TEST_CASE("test detection model predict function", "[detection_model_predict]") {
    REQUIRE (test_detection_model_predict() == StatusCode::Success);
}


TEST_CASE("test release detection result", "[release_detection_result]") {
    REQUIRE_NOTHROW(test_release_detection_result1());
    REQUIRE_NOTHROW(test_release_detection_result2());
}

TEST_CASE("test release detection model function", "[release_detection_model]") {
    REQUIRE_NOTHROW(test_release_detection_model());
}