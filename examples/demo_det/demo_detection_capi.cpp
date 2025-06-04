//
// Created by aichao on 2025/2/24.
//

#include <iostream>
#include <chrono>

#include "capi/utils/md_image_capi.h"
#include "capi/vision/detection/detection_capi.h"

int main() {
    MDStatusCode ret;
    MDModel model;
    if (ret = md_create_detection_model(&model, "../../test_data/test_models/yolo11n.onnx", 8); ret) {
        std::cout << ret << std::endl;
        return ret;
    }
    if (constexpr MDSize size = {640, 640}; (ret = md_set_detection_input_size(&model, size)) != 0) {
        std::cout << ret << std::endl;
        return ret;
    }
    const std::chrono::system_clock::time_point start = std::chrono::system_clock::now();
    auto im = md_read_image("../../test_data/test_images/test_detection.jpg");
    MDDetectionResults result;
    if ((ret = md_detection_predict(&model, &im, &result)) != 0) {
        std::cout << ret << std::endl;
        return ret;
    }
    md_draw_detection_result(&im, &result, 0.3, "../../test_data/msyh.ttc", 14, 0.5, 1);
    const std::chrono::duration<double> diff = std::chrono::system_clock::now() - start;
    std::cout << "duration cost: " << diff.count() << "s" << std::endl;
    md_show_image(&im);
    md_print_detection_result(&result);
    md_free_detection_result(&result);
    md_free_image(&im);
    md_free_detection_model(&model);
    return ret;
}
