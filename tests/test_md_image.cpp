//
// Created by aichao on 2025/3/18.
//

#include <iostream>

#include <filesystem>
#include <fstream>

#include "tests/utils.h"
#include <catch2/catch_test_macros.hpp>
#include "capi/utils/md_image_capi.h"


bool read_image() {
    const std::filesystem::path image_path = TEST_DATA_DIR / "test_images" / "test_person.jpg";
    const std::string image_path_str = image_path.string();
    const MDImage image = md_read_image(image_path_str.c_str());
    return image.data != nullptr && image.height == 675 && image.width == 900;
}

bool from_base64_str() {
    const std::filesystem::path image_path = TEST_DATA_DIR/ "test_images" / "test_base64_image.txt";;
    const std::string image_path_str = image_path.string();
    std::ifstream file(image_path_str);
    if (!file.is_open()) {
        return false;
    }
    const std::string base64_str((std::istreambuf_iterator(file)), std::istreambuf_iterator<char>());
    file.close();
    const MDImage image = md_from_base64_str(base64_str.c_str());
    return image.data != nullptr && image.height == 184 && image.width == 192;
}


TEST_CASE("ModelDeploy Image", "[md_image]") {
    REQUIRE(read_image()==true);
    REQUIRE(from_base64_str()==true);
}
