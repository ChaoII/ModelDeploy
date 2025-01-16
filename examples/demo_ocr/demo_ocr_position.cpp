//
// Created by AC on 2025-01-13.
//
#include <iostream>
#include <chrono>
#include "src/utils/utils_capi.h"
#include "src/ocr/ocr_capi.h"
#include <fstream>
#include <sstream>

#ifdef WIN32

#include <windows.h>

#endif

#include <vector>

//字符串分割函数
std::vector<std::string> split(std::string str, const std::string& pattern) {
    std::string::size_type pos;
    std::vector<std::string> result;
    str += pattern; //扩展字符串以方便操作
    int size = str.size();
    for (int i = 0; i < size; i++) {
        pos = str.find(pattern, i);
        if (pos < size) {
            std::string s = str.substr(i, pos - i);
            result.push_back(s);
            i = pos + pattern.size() - 1;
        }
    }
    return result;
}


int main(int argc, char** argv) {
#ifdef WIN32
    SetConsoleOutputCP(CP_UTF8);
    SetConsoleCP(CP_UTF8);
#endif
    MDStatusCode ret;

    std::string parameter_file = argv[1];
    std::fstream in(parameter_file);
    if (!in) {
        std::cout << "Error: parameter file not found" << std::endl;
        return -1;
    }


    // 创建模型句柄
    MDModel model;
    MDOCRModelParameters ocr_parameters = {
        "../tests/models/ocr",
        "../tests/key.txt",
        8,
        PaddlePaddle,
        960,
        0.3,
        0.6,
        1.5,
        "slow",
        0,
        4
    };
    if ((ret = md_create_ocr_model(&model, &ocr_parameters)) != 0) {
        std::cout << ret << std::endl;
        return ret;
    }
    // 获取文本目标位置
    //    MDOCRResults results;
    //    if ((ret = md_ocr_model_predict(&model, &image, &results)) != 0) {
    //        std::cout << ret << std::endl;
    //        return ret;
    //    }
    MDColor color = {0, 0, 255};
    //    std::chrono::system_clock::time_point start = std::chrono::system_clock::now();
    //    md_draw_ocr_result(&image, &results, "../tests/msyh.ttc", 15, &color, 0.5, 1);
    //    std::chrono::duration<double> diff = std::chrono::system_clock::now() - start;
    //    std::cout << "cost: " << diff.count() << std::endl;
    //    md_print_ocr_result(&results);
    //    md_free_ocr_result(&results);

    std::string line;
    while (std::getline(in, line)) {
        auto parameters = split(line, " ");
        if (parameters.size() != 2) {
            std::cerr << "Error: parameter file format error" << std::endl;
            return -1;
        }
        auto image_path = parameters[0];
        auto button_name = parameters[1];
        auto image = md_read_image(image_path.c_str());
        auto text = button_name.c_str();
        auto rect = md_get_text_position(&model, &image, text);
        // 打印文本信息
        md_print_rect(&rect);
        // 先裁剪再绘制，不然image是指针传递，在绘制时会修改原始image
        if (rect.width > 0 && rect.height > 0) {
            auto roi = md_crop_image(&image, &rect);
            // 在原始画面上绘制文本结果
            // 判断按钮是否可用
            auto enable = md_get_button_enable_status(&roi, 50, 0.05);

            std::stringstream ss;
            ss << (std::string(text) + "-" + (enable ? "enable" : "disable"))
                << "-[ " << "x: " << rect.x << " y: " << rect.y << " ]";

            md_draw_text(&image, &rect,
                         ss.str().c_str(),
                         "../tests/msyh.ttc",
                         15,
                         &color, 0.5);

            //        md_show_image(&roi);
            // 释放资源
            md_free_image(&roi);
        }

        // 显示原始画面
        md_show_image(&image);
        // 显示目标文本所在画面
        md_free_image(&image);
    }


    md_free_ocr_model(&model);
    return ret;
}
