//
// Created by AC on 2024-12-23.
//


#include "../src/detection_capi.h"
#include "../src/utils.h"
#include <catch2/catch_test_macros.hpp>
#include <filesystem>

MDStatusCode test_create_detection_model(const char *model_path, MDModelFormat format) {
    MDModel model;
    return md_create_detection_model(&model, model_path, 8, format);
}

MDStatusCode test_set_detection_input_size(const char *model_path, MDModelFormat format) {
    MDModel model;
    md_create_detection_model(&model, model_path, 8, format);
    return md_set_detection_input_size(&model, {1440, 1440});
}


MDStatusCode test_detection_model_predict() {
    MDModel model;
    md_create_detection_model(&model, "../../tests/models/best.onnx", 8);
    md_set_detection_input_size(&model, {1440, 1440});
    auto im = md_read_image("../../tests/test_images/test_detection.png");
    MDDetectionResults result;
    return md_detection_predict(&model, im, &result);
}

void test_release_detection_result1() {
    MDDetectionResults result{nullptr, 0};
    md_free_detection_result(&result);
}


void test_release_detection_result2() {
    MDDetectionResults result;
    result.size = 2;
    result.data = (MDDetectionResult *) malloc(sizeof(MDDetectionResult) * result.size);
    result.data[0].box = {0, 0, 100, 100};
    result.data[1].box = {100, 100, 200, 200};
    result.data[0].label_id = 1;
    result.data[1].label_id = 2;
    result.data[0].score = 0.5;
    result.data[1].score = 0.6;
    md_free_detection_result(&result);
}

void test_release_detection_model() {
    MDModel model;
    md_create_detection_model(&model, "../../tests/models/best.onnx", 8);
    return md_free_detection_model(&model);
}

TEST_CASE("test create detection model function", "[create_detection_model]") {
    REQUIRE (test_create_detection_model("../../tests/models/best.onnx", MDModelFormat::ONNX) == MDStatusCode::Success);
    REQUIRE (test_create_detection_model("../../tests/models/ppyoloe", MDModelFormat::PaddlePaddle) ==
             MDStatusCode::Success);
    REQUIRE (test_create_detection_model("../../tests/models/picodet_lcnet", MDModelFormat::PaddlePaddle) ==
             MDStatusCode::Success);
}

TEST_CASE("test set detection input size function", "[set_detection_input_size]") {
    REQUIRE (test_set_detection_input_size("../../tests/models/best.onnx", MDModelFormat::ONNX) == MDStatusCode::Success);
    REQUIRE (test_set_detection_input_size("../../tests/models/ppyoloe", MDModelFormat::PaddlePaddle) ==
             MDStatusCode::CallError);
}

TEST_CASE("test detection model predict function", "[detection_model_predict]") {
    REQUIRE (test_detection_model_predict() == MDStatusCode::Success);
}


TEST_CASE("test release detection result", "[release_detection_result]") {
    REQUIRE_NOTHROW(test_release_detection_result1());
    REQUIRE_NOTHROW(test_release_detection_result2());
}

TEST_CASE("test release detection model function", "[release_detection_model]") {
    REQUIRE_NOTHROW(test_release_detection_model());
}