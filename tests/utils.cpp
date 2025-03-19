#include "tests/utils.h"
#include <iostream>

std::filesystem::path get_test_data_path(){
    const char* test_data_dir_env = std::getenv("TEST_DATA_DIR");
    std::filesystem::path test_data_path;
    if (test_data_dir_env && *test_data_dir_env) {
        test_data_path = std::filesystem::path(test_data_dir_env);
        if (std::filesystem::exists(test_data_path)) {
            return test_data_path;
        }
    }
    const auto current_path = std::filesystem::current_path();
    const auto dir_name = current_path.filename().string();
    if (dir_name == "tests") {
        test_data_path = current_path.parent_path().parent_path() / "test_data";
    }
    else if (dir_name == "build") {
        test_data_path = current_path.parent_path() / "test_data";
    }
    else {
        test_data_path = current_path / "test_data";
    }
    return test_data_path;
}