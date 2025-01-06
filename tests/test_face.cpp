//
// Created by AC on 2024-12-27.
//

#include "../src/face/face_capi.h"
#include "src/utils/utils_capi.h"
#include <catch2/catch_test_macros.hpp>

MDStatusCode test_md_create_face_model() {
    MDModel model;
    return md_create_face_model(&model, "../../tests/models/seetaface", MD_MASK, 1);
}


TEST_CASE("test create face model function", "[create_face_model]") {
    REQUIRE(test_md_create_face_model() == MDStatusCode::Success);
}


