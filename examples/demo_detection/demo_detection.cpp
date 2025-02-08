//
// Created by AC on 2025-01-13.
//
#include <iostream>
#include <chrono>

#include "csrc/model_deploy.h"

int main() {

    MDStatusCode ret;
    MDModel model;
    if ((ret = md_create_detection_model(&model, "../tests/models/best.onnx", 4)) != 0) {
        std::cout << ret << std::endl;
        return ret;
    }
    MDSize size = {1440, 1440};
    if ((ret = md_set_detection_input_size(&model, size)) != 0) {
        std::cout << ret << std::endl;
        return ret;
    }

    auto im = md_read_image("../tests/test_images/test_detection.png");

    MDDetectionResults result;
    if ((ret = md_detection_predict(&model, &im, &result)) != 0) {
        std::cout << ret << std::endl;
        return ret;
    }
    std::chrono::system_clock::time_point start = std::chrono::system_clock::now();
    md_draw_detection_result(&im, &result, "../tests/msyh.ttc", 20, 0.5, 1);
    std::chrono::duration<double> diff = std::chrono::system_clock::now() - start;
    std::cout << "cost: " << diff.count() << std::endl;
//    md_show_image(&im);
    md_print_detection_result(&result);
    md_free_detection_result(&result);

    md_free_image(&im);
    md_free_detection_model(&model);
    return ret;
    return 0;
}