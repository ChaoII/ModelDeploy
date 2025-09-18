//
// Created by aichao on 2025/3/18.
//
#pragma once

#include <filesystem>
#include "capi/utils/md_image_capi.h"
#include "csrc/vision/common/image_data.h"

std::filesystem::path get_test_data_path();

#define TEST_DATA_DIR [](){return get_test_data_path();}()


void print_md_image_pixels(const MDImage* image, int rows = 5, int cols = 5);

// C++ ImageData 版本
void print_imagedata_pixels(const modeldeploy::ImageData& img, int rows = 5, int cols = 5);

void compare_cpp_c_image(const modeldeploy::ImageData& img, const MDImage* c_image, int sample_count = 20);
