//
// Created by aichao on 2025/2/24.
//

#include <iostream>
#include <chrono>

#include "capi/utils/md_image_capi.h"
#include "capi/vision/classification/classification_capi.h"

int main() {
    MDStatusCode ret;
    MDModel model;
    if (ret = md_create_classification_model(&model, "../../test_data/test_models/yolov5n-cls.onnx", 8); ret) {
        std::cout << ret << std::endl;
        return ret;
    }
    if (constexpr MDSize size = {224, 224}; (ret = md_set_classification_input_size(&model, size)) != 0) {
        std::cout << ret << std::endl;
        return ret;
    }
    const std::chrono::system_clock::time_point start = std::chrono::system_clock::now();
    auto im = md_read_image("../../test_data/test_images/test_face.jpg");
    MDClassificationResults result;
    if ((ret = md_classification_predict(&model, &im, &result)) != 0) {
        std::cout << ret << std::endl;
        return ret;
    }
    md_draw_classification_result(&im, &result, 1, 0.5, "../../test_data/msyh.ttc", 14, 0.5, 1);
    const std::chrono::duration<double> diff = std::chrono::system_clock::now() - start;
    std::cout << "duration cost: " << diff.count() << "s" << std::endl;
    md_show_image(&im);
    md_print_classification_result(&result);
    md_free_classification_result(&result);
    md_free_image(&im);
    md_free_classification_model(&model);
    return ret;
}
