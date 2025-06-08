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
#endif
    MDStatusCode ret;
    //简单百宝箱
    MDModel model;
    MDOCRModelParameters ocr_parameters = {
        "../../test_data/test_models/ocr/ppocrv5_mobile/det_infer.onnx",
        "../../test_data/test_models/ocr/ppocrv5_mobile/cls_infer.onnx",
        "../../test_data/test_models/ocr/ppocrv5_mobile/rec_infer.onnx",
        "../../test_data/ppocrv5_dict.txt",
        1,
        8,
        ONNX,
        1920,
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
    MDImage image = md_read_image("../../test_data/test_images/test_ocr9.jpg");
    MDOCRResults results;
    if ((ret = md_ocr_model_predict(&model, &image, &results)) != 0) {
        std::cout << ret << std::endl;
        return ret;
    }
    const std::chrono::system_clock::time_point start = std::chrono::system_clock::now();
    md_draw_ocr_result(&image, &results, "../../test_data/msyh.ttc", 12, 0.1, 1);
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

