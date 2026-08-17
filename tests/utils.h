//
// Created by aichao on 2025/3/18.
//
#pragma once

#include <filesystem>
#include "csrc/vision/common/image_data.h"
#include "csrc/vision.h"

std::filesystem::path get_test_data_path();

#define TEST_DATA_DIR [](){return get_test_data_path();}()

// C++ ImageData �汾
void print_imagedata_pixels(const modeldeploy::vision::ImageData& img, int rows = 5, int cols = 5);
