//
// Created by AC on 2024-12-16.
//
#include "../src/ocr/ocr_capi.h"
#include "../src/utils/utils.h"
#include <catch2/catch_test_macros.hpp>


bool almost_equal_rect(MDRect rect1, MDRect rect2) {
    return rect1.x == rect2.x && rect1.y == rect2.y && rect1.width == rect2.width && rect1.height == rect2.height;
}

MDStatusCode test_create_ocr_model(MDOCRModelParameters *parameters) {
    MDModel model;
    return md_create_ocr_model(&model, parameters);
}


MDStatusCode test_ocr_model_predict(MDOCRModelParameters *parameters) {
    MDModel model;
    md_create_ocr_model(&model, parameters);
    auto im = md_read_image("../../tests/test_images/test_ocr.png");
    MDOCRResults result;
    return md_ocr_model_predict(&model, &im, &result);
}

void test_release_ocr_result1() {
    MDOCRResults result{nullptr, 0};
    md_free_ocr_result(&result);
}


MDRect test_get_text_position(const char *text) {
    MDModel model;
    MDOCRModelParameters parameters{
            "../../tests/models/ocr",
            "../../tests/key.txt", 8, MDModelFormat::PaddlePaddle,
            960, 0.3, 0.6,
            1.5, "slow",
            0, 8
    };
md_create_ocr_model(&model, &parameters);
    auto im = md_read_image("../../tests/test_images/test_ocr.png");
    return md_get_text_position(&model, &im, text);
}

void test_release_ocr_result2() {
    MDOCRResults result;
    result.size = 2;
    result.data = (MDOCRResult *) malloc(sizeof(MDOCRResult) * result.size);
    MDPolygon polygon0;
    polygon0.size = 4;
    polygon0.data = (MDPoint *) malloc(sizeof(MDPoint) * polygon0.size);
    polygon0.data[0] = {0, 0};
    polygon0.data[1] = {100, 0};
    polygon0.data[2] = {100, 100};
    polygon0.data[3] = {0, 100};

    MDPolygon polygon1;
    polygon1.size = 4;
    polygon1.data = (MDPoint *) malloc(sizeof(MDPoint) * polygon0.size);
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
md_free_ocr_result(&result);
}

void test_release_ocr_model() {
    MDModel model;
    MDOCRModelParameters parameters{
            "../../tests/models/ocr",
            "../../tests/key.txt", 8, MDModelFormat::PaddlePaddle,
            960, 0.3, 0.6,
            1.5, "slow",
            0, 8
    };
md_create_ocr_model(&model, &parameters);
    return md_free_ocr_model(&model);
}

TEST_CASE("test create ocr model function", "[create_ocr_model]") {
    MDOCRModelParameters parameters1{
            "../../tests/models/ocr",
            "../../tests/key.txt", 8, MDModelFormat::PaddlePaddle,
            960, 0.3, 0.6,
            1.5, "slow",
            0, 8
    };
    MDOCRModelParameters parameters2{
            "../../tests/models/ocr",
            "../../tests/key.txt", 1, MDModelFormat::PaddlePaddle,
            960, 0.3, 0.6,
            1.5, "fast",
            1, 4
    };

    REQUIRE(test_create_ocr_model(&parameters1) == MDStatusCode::Success);
    REQUIRE(test_create_ocr_model(&parameters2) == MDStatusCode::Success);
}


TEST_CASE("test ocr model predict function", "[ocr_model_predict]") {
    MDOCRModelParameters parameters{
            "../../tests/models/ocr",
            "../../tests/key.txt", 8, MDModelFormat::PaddlePaddle,
            960, 0.3, 0.6,
            1.5, "slow",
            0, 8};
    REQUIRE (test_ocr_model_predict(&parameters) == MDStatusCode::Success);
}

TEST_CASE("test get text position", "[get_text_position]") {
    REQUIRE(almost_equal_rect(test_get_text_position("暂停测试"), {665, 808, 74, 30}));
    REQUIRE(almost_equal_rect(test_get_text_position("开始测试"), {775, 806, 97, 30}));
}

TEST_CASE("test release ocr result", "[release_detection_result]") {
    REQUIRE_NOTHROW (test_release_ocr_result1());
    REQUIRE_NOTHROW (test_release_ocr_result2());
}

TEST_CASE("test release ocr model function", "[release_detection_model]") {
    REQUIRE_NOTHROW (test_release_ocr_model());
}