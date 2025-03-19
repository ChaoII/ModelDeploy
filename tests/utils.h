//
// Created by aichao on 2025/3/18.
//
#pragma once

#include <filesystem>

std::filesystem::path get_test_data_path();

#define TEST_DATA_DIR [](){return get_test_data_path();}()