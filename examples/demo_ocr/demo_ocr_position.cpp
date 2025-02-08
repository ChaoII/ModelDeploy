//
// Created by AC on 2025-01-13.
//

#include <iostream>
#include <chrono>
#include <fstream>
#include <sstream>
#include <vector>
#ifdef WIN32
#include <windows.h>
#endif
#include "csrc/model_deploy.h"



int main() {
#ifdef WIN32
    SetConsoleOutputCP(CP_UTF8);
    SetConsoleCP(CP_UTF8);
#endif
    MDStatusCode ret;

    // 创建模型句柄
    MDModel model;
    MDOCRModelParameters ocr_parameters = {
        "../tests/models/ocr",
        "../tests/key.txt",
        4,
        PaddlePaddle,
        1440,
        0.3,
        0.6,
        1.5,
        "fast",
        0,
        4
    };
    if ((ret = md_create_ocr_model(&model, &ocr_parameters)) != 0) {
        std::cout << ret << std::endl;
        return ret;
    }

    const int loop_count = 10;
    const char* text = "暂停测试";
    const MDColor color = {0, 0, 255};
    MDImage image = md_read_image("../tests/test_images/test_ocr.png");
    auto start_time = std::chrono::steady_clock::now();
    for (int i = 0; i < loop_count; i++) {
        auto rect = md_get_text_position(&model, &image, text);
        // 打印文本信息
        md_print_rect(&rect);
        // 先裁剪再绘制，不然image是指针传递，在绘制时会修改原始image
        if (rect.width > 0 && rect.height > 0) {
            // auto roi = md_crop_image(&image, &rect);
            // 在原始画面上绘制文本结果
            // 判断按钮是否可用
            // auto enable = md_get_button_enable_status(&roi, 50, 0.05);
            // std::stringstream ss;
            // ss << (std::string(text) + "-" + (enable ? "enable" : "disable"))
            //     << "-[ " << "x: " << rect.x << " y: " << rect.y << " ]";
            // md_draw_text(&image, &rect,
            //              ss.str().c_str(),
            //              "../tests/msyh.ttc",
            //              15,
            //              &color, 0.5);
            //        md_show_image(&roi);
            // 释放资源
            // md_free_image(&roi);
        }
    }
    auto end_time = std::chrono::steady_clock::now();

    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    std::cout << "Time taken: " << duration.count() / (float)loop_count << " ms" << std::endl;


    // 显示原始画面
    // md_show_image(&image);
    // 显示目标文本所在画面
    md_free_image(&image);

    md_free_ocr_model(&model);
    return ret;
}
