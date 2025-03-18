//
// Created by aichao on 2025/3/18.
//
#pragma once

#include <filesystem>

std::filesystem::path get_test_data_path();


const std::filesystem::path TEST_DATA_DIR = get_test_data_path();
