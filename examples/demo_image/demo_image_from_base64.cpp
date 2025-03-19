//
// Created by aichao on 2025/3/18.
//

#include <fstream>
#include <iostream>
#include <filesystem>
#include "capi/utils/md_image_capi.h"

int main(int argc, char** argv) {
    //读取base64_image.txt文件

    auto file_path = "../test_data/test_images/test_base64_image.txt";
    std::ifstream file(file_path);
    if (!file.is_open()) {
        std::cerr << "Failed to open file: " << file_path << std::endl;
    }
    std::string base64_str((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
    file.close();
    MDImage image = md_from_base64_str(base64_str.c_str());
    md_save_image(&image, "s.jpg");
    md_show_image(&image);
}
