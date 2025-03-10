//
// Created by aichao on 2025/2/24.
//

#include <iostream>
#include <chrono>

#include "capi/vision/detection/detection_capi.h"
#include "capi/utils/md_image_capi.h"

int main() {
    MDStatusCode ret;
    MDModel model;
    if (ret = md_create_detection_model(&model, "../tests/test_models/best.onnx", 8); ret) {
        std::cout << ret << std::endl;
        return ret;
    }
    if (constexpr MDSize size = {1440, 1440}; (ret = md_set_detection_input_size(&model, size)) != 0) {
        std::cout << ret << std::endl;
        return ret;
    }
    const std::chrono::system_clock::time_point start = std::chrono::system_clock::now();
    auto im = md_read_image("../tests/test_images/test_detection.png");
    MDDetectionResults result;
    if ((ret = md_detection_predict(&model, &im, &result)) != 0) {
        std::cout << ret << std::endl;
        return ret;
    }
    md_draw_detection_result(&im, &result, "../tests/msyh.ttc", 20, 0.5, 1);
    const std::chrono::duration<double> diff = std::chrono::system_clock::now() - start;
    std::cout << "duration cost: " << diff.count() << "s" << std::endl;
    // md_show_image(&im);
    // md_print_detection_result(&result);
    md_free_detection_result(&result);

    md_free_image(&im);
    md_free_detection_model(&model);
    return ret;
    return 0;
}
