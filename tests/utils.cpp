#include "tests/utils.h"
#include <iostream>
#include <random>

std::filesystem::path get_test_data_path() {
    const char* test_data_dir_env = std::getenv("TEST_DATA_DIR");
    if (test_data_dir_env) {
        std::cout << "TEST_DATA_DIR: " << test_data_dir_env << std::endl;
    } else {
        std::cout << "TEST_DATA_DIR: (not set)" << std::endl;
    }
    std::filesystem::path test_data_path;
    if (test_data_dir_env && *test_data_dir_env) {
        test_data_path = std::filesystem::path(std::string(test_data_dir_env) + "/test_data");
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

// C++ ImageData 版本
void print_imagedata_pixels(const modeldeploy::vision::ImageData& img, const int rows, const int cols) {
    const int max_r = std::min(rows, img.height());
    const int max_c = std::min(cols, img.width());
    std::cout << "ImageData pixels (decimal / hex):" << std::endl;
    for (int r = 0; r < max_r; r++) {
        for (int c = 0; c < max_c; c++) {
            std::cout << "(";
            for (int ch = 0; ch < img.channels(); ch++) {
                // 获取像素值，假设 img.plane(0).data 是一个指向图像数据的指针
                const uint8_t val = img.plane(0).data[r * img.width() * img.channels() + c * img.channels() + ch];
                std::cout << static_cast<int>(val) << "/" << std::hex << static_cast<int>(val) << std::dec;
                if (ch != img.channels() - 1) std::cout << ",";
            }
            std::cout << ") ";
        }
        std::cout << std::endl;
    }
}
