//
// Created by AC on 2024-12-16.
//
#include "../src/ocr_capi.h"
#include "../src/utils.h"
#include <catch2/catch_test_macros.hpp>


bool almost_equal_rect(WRect rect1, WRect rect2) {
    return rect1.x == rect2.x && rect1.y == rect2.y && rect1.width == rect2.width && rect1.height == rect2.height;
}

StatusCode test_create_ocr_model(OCRModelParameters *parameters) {
    WModel model;
    return create_ocr_model(&model, parameters);
}


StatusCode test_ocr_model_predict(OCRModelParameters *parameters) {
    WModel model;
    create_ocr_model(&model, parameters);
    auto im = read_image("test_ocr.png");
    WOCRResults result;
    return ocr_model_predict(&model, im, &result, 1, {255, 255, 0}, 0.5, 1);
}

void test_release_ocr_result1() {
    WOCRResults result{nullptr, 0};
    free_ocr_result(&result);
}


WRect test_get_text_position(const char *text) {
    WModel model;
    OCRModelParameters parameters{
            "models",
            "key.txt", 8, ModelFormat::PaddlePaddle,
            960, 0.3, 0.6,
            1.5, "slow",
            0, 8
    };
    create_ocr_model(&model, &parameters);
    WImage *im = read_image("test_ocr.png");
    return get_text_position(&model, im, text);
}

void test_release_ocr_result2() {
    WOCRResults result;
    result.size = 2;
    result.data = (WOCRResult *) malloc(sizeof(WOCRResult) * result.size);
    WPolygon polygon0;
    polygon0.size = 4;
    polygon0.data = (WPoint *) malloc(sizeof(WPoint) * polygon0.size);
    polygon0.data[0] = {0, 0};
    polygon0.data[1] = {100, 0};
    polygon0.data[2] = {100, 100};
    polygon0.data[3] = {0, 100};

    WPolygon polygon1;
    polygon1.size = 4;
    polygon1.data = (WPoint *) malloc(sizeof(WPoint) * polygon0.size);
    polygon1.data[0] = {10, 10};
    polygon1.data[1] = {120, 10};
    polygon1.data[2] = {120, 150};
    polygon1.data[3] = {10, 150};

    result.data[0].box = polygon0;
    result.data[1].box = polygon1;
    result.data[0].text = (char *) malloc(10);
    memcpy(result.data[0].text, "hello", 5);
    result.data[1].text = (char *) malloc(6);
    memcpy(result.data[1].text, "world", 5);
    result.data[0].score = 0.5;
    result.data[1].score = 0.6;
    free_ocr_result(&result);
}

void test_release_ocr_model() {
    WModel model;
    OCRModelParameters parameters{
            "models",
            "key.txt", 8, ModelFormat::PaddlePaddle,
            960, 0.3, 0.6,
            1.5, "slow",
            0, 8
    };
    create_ocr_model(&model, &parameters);
    return free_ocr_model(&model);
}

TEST_CASE("test create ocr model function", "[create_ocr_model]") {
    OCRModelParameters parameters1{
            "models",
            "key.txt", 8, ModelFormat::PaddlePaddle,
            960, 0.3, 0.6,
            1.5, "slow",
            0, 8
    };
    OCRModelParameters parameters2{
            "models",
            "key.txt", 1, ModelFormat::PaddlePaddle,
            960, 0.3, 0.6,
            1.5, "fast",
            1, 4
    };

    REQUIRE(test_create_ocr_model(&parameters1) == StatusCode::Success);
    REQUIRE(test_create_ocr_model(&parameters2) == StatusCode::Success);
}


TEST_CASE("test ocr model predict function", "[ocr_model_predict]") {
    OCRModelParameters parameters{
            "models",
            "key.txt", 1, ModelFormat::PaddlePaddle,
            960, 0.3, 0.6,
            1.5, "fast",
            1, 4
    };
    REQUIRE (test_ocr_model_predict(&parameters) == StatusCode::Success);
}

TEST_CASE("test get text position", "[get_text_position]") {
    REQUIRE(almost_equal_rect(test_get_text_position("暂停测试"), {655, 808, 74, 30}));
    REQUIRE(almost_equal_rect(test_get_text_position("暂停测试"), {775, 806, 97, 30}));
}

TEST_CASE("test release ocr result", "[release_detection_result]") {
    REQUIRE_NOTHROW (test_release_ocr_result1());
    REQUIRE_NOTHROW (test_release_ocr_result2());
}

TEST_CASE("test release ocr model function", "[release_detection_model]") {
    REQUIRE_NOTHROW (test_release_ocr_model());
}