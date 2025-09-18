#include "tests/utils.h"
#include <iostream>
#include <random>

std::filesystem::path get_test_data_path() {
    const char* test_data_dir_env = std::getenv("TEST_DATA_DIR");
    std::cout << "TEST_DATA_DIR: " << test_data_dir_env << std::endl;
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

void print_md_image_pixels(const MDImage* image, const int rows, const int cols) {
    if (!image) return;

    const int max_r = std::min(rows, image->height);
    const int max_c = std::min(cols, image->width);

    std::cout << "MDImage pixels (decimal / hex):" << std::endl;
    for (int r = 0; r < max_r; r++) {
        for (int c = 0; c < max_c; c++) {
            std::cout << "(";
            for (int ch = 0; ch < image->channels; ch++) {
                const uint8_t val = image->data[r * image->width * image->channels + c * image->channels + ch];
                std::cout << static_cast<int>(val) << "/" << std::hex << static_cast<int>(val) << std::dec;
                if (ch != image->channels - 1) std::cout << ",";
            }
            std::cout << ") ";
        }
        std::cout << std::endl;
    }
}

// C++ ImageData 版本
void print_imagedata_pixels(const modeldeploy::ImageData& img, const int rows, const int cols) {
    const int max_r = std::min(rows, img.height());
    const int max_c = std::min(cols, img.width());
    std::cout << "ImageData pixels (decimal / hex):" << std::endl;
    for (int r = 0; r < max_r; r++) {
        for (int c = 0; c < max_c; c++) {
            std::cout << "(";
            for (int ch = 0; ch < img.channels(); ch++) {
                // 获取像素值，假设 img.data() 是一个指向图像数据的指针
                const uint8_t val = img.data()[r * img.width() * img.channels() + c * img.channels() + ch];
                std::cout << static_cast<int>(val) << "/" << std::hex << static_cast<int>(val) << std::dec;
                if (ch != img.channels() - 1) std::cout << ",";
            }
            std::cout << ") ";
        }
        std::cout << std::endl;
    }
}


void compare_cpp_c_image(const modeldeploy::ImageData& img, const MDImage* c_image, const int sample_count) {
    if (!c_image || img.empty()) return;
    if (img.height() != c_image->height || img.width() != c_image->width || img.channels() != c_image->channels) {
        std::cerr << "Image dimension/channel mismatch!\n";
        return;
    }
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution dis_row(0, img.height() - 1);
    std::uniform_int_distribution dis_col(0, img.width() - 1);
    std::cout << "Comparing " << sample_count << " random pixels (decimal / hex):\n";
    for (int i = 0; i < sample_count; ++i) {
        const int r = dis_row(gen); // 随机行
        const int c = dis_col(gen); // 随机列
        std::cout << "Pixel (" << r << "," << c << "):\n";
        // 获取像素数据函数，用于减少重复代码
        auto print_pixel = [](const uint8_t* px_cpp, const unsigned char* px_c, int channels) {
            for (int ch = 0; ch < channels; ++ch) {
                std::cout << static_cast<int>(px_cpp[ch]) << "/" << std::hex << static_cast<int>(px_cpp[ch]) <<
                    std::dec;
                if (ch != channels - 1) std::cout << ",";
            }
            std::cout << ")\n";
            for (int ch = 0; ch < channels; ++ch) {
                std::cout << static_cast<int>(px_c[ch]) << "/" << std::hex << static_cast<int>(px_c[ch]) << std::dec;
                if (ch != channels - 1) std::cout << ",";
            }
            std::cout << ")\n";
        };

        // 处理3通道图像
        if (img.channels() == 3) {
            const uint8_t* px_cpp = img.data() + (r * img.width() + c) * img.channels();
            const uint8_t* px_c = c_image->data + (r * c_image->width + c) * c_image->channels;
            std::cout << "  C++: (";
            print_pixel(px_cpp, px_c, 3); // 调用打印函数
        }
        // 处理1通道灰度图像
        else if (img.channels() == 1) {
            const uint8_t px_cpp = img.data()[r * img.width() + c];
            const uint8_t px_c = c_image->data[r * c_image->width + c];
            std::cout << "  C++: " << static_cast<int>(px_cpp) << "/"
                << std::hex << static_cast<int>(px_cpp) << std::dec << std::endl;
            std::cout << "  C  : " << static_cast<int>(px_c) << "/"
                << std::hex << static_cast<int>(px_c) << std::dec << std::endl;
        }
        // 处理4通道图像 (RGBA)
        else if (img.channels() == 4) {
            const uint8_t* px_cpp = img.data() + (r * img.width() + c) * img.channels();
            const uint8_t* px_c = c_image->data + (r * c_image->width + c) * c_image->channels;
            std::cout << "  C++: (";
            print_pixel(px_cpp, px_c, 4); // 调用打印函数
        }
        std::cout << "-------------------------\n";
    }
}
