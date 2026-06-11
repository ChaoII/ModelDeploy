//
// Created by aichao on 2025/3/18.
//

#include <iostream>

#include <filesystem>
#include <fstream>

#include "tests/utils.h"
#include "capi/utils/md_image_capi.h"
#include <catch2/catch_test_macros.hpp>

#include <opencv2/core.hpp>

static bool test_images_available() {
    auto dir = get_test_data_path() / "test_images";
    return std::filesystem::exists(dir / "test_person.jpg") &&
           std::filesystem::exists(dir / "test_base64_image.txt");
}

static bool read_image() {
    const auto dir = get_test_data_path() / "test_images";
    const auto path = dir / "test_person.jpg";
    const MDImage image = md_read_image(path.string().c_str());
    return image.data != nullptr && image.height == 675 && image.width == 900;
}

static bool from_base64_str() {
    const auto dir = get_test_data_path() / "test_images";
    const auto path = dir / "test_base64_image.txt";
    std::ifstream file(path);
    if (!file.is_open()) return false;
    const std::string base64_str((std::istreambuf_iterator(file)), std::istreambuf_iterator<char>());
    file.close();
    const MDImage image = md_from_base64_str(base64_str.c_str());
    return image.data != nullptr && image.height == 184 && image.width == 192;
}


TEST_CASE("ModelDeploy Image", "[md_image]") {
    if (!test_images_available()) {
        FAIL("Test images not found. Set TEST_DATA_DIR environment variable.");
    }
    REQUIRE(read_image());
    REQUIRE(from_base64_str());
}
