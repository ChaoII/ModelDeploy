//
// Created by AC on 2024-12-16.
//

#include "../wrzs_capi.h"
#include "../utils.h"
#include <catch2/catch_test_macros.hpp>
#include <opencv2/opencv.hpp>


int test_get_template_position() {
    auto shot_mat = cv::imread("../test_images/shot_image.png");
    auto temp_mat = cv::imread("../test_images/temp1.png");
    auto shot_image = mat_to_wimage(shot_mat);
    auto temp_image = mat_to_wimage(temp_mat);
    auto rect = get_template_position(shot_image, temp_image);
}

TEST_CASE("text rec1", "[text_rec]") {
REQUIRE (test_get_template_position()

== 0);
}