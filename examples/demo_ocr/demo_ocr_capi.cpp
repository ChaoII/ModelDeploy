//
// Created by aichao on 2025/2/26.
//

#include <iostream>
#include <chrono>
#include <fstream>
#ifdef WIN32
#include <windows.h>
#endif
#include "capi/utils/md_image_capi.h"
#include "capi/vision/ocr/ocr_capi.h"

int main(int argc, char** argv) {
#ifdef WIN32
    SetConsoleOutputCP(CP_UTF8);
    SetConsoleCP(CP_UTF8);
#endif
    MDStatusCode ret;
    //简单百宝箱
    MDModel model;
    MDOCRModelParameters ocr_parameters = {
        "../tests/test_models/ocr/",
        "../tests/key.txt",
        8,
        ONNX,
        960,
        0.3,
        0.6,
        1.5,
        "slow",
        0,
        8
    };
    if ((ret = md_create_ocr_model(&model, &ocr_parameters)) != 0) {
        std::cout << ret << std::endl;
        return ret;
    }
    MDImage image = md_read_image("../tests/test_images/test_ocr1.png");
    MDOCRResults results;
    if ((ret = md_ocr_model_predict(&model, &image, &results)) != 0) {
        std::cout << ret << std::endl;
        return ret;
    }
    constexpr MDColor color = {255, 0, 255};
    const std::chrono::system_clock::time_point start = std::chrono::system_clock::now();
    md_draw_ocr_result(&image, &results, "../tests/msyh.ttc", 15, &color, 0.5, 1);
    const std::chrono::duration<double> diff = std::chrono::system_clock::now() - start;
    std::cout << "cost: " << diff.count() << std::endl;
    md_print_ocr_result(&results);
    md_free_ocr_result(&results);
    md_show_image(&image);
    // 释放内存
    md_free_image(&image);
    md_free_ocr_model(&model);
    return ret;
}

