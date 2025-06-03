//
// Created by aichao on 2025/5/26.
//

#include <iostream>
#include <chrono>

#include "capi/utils/md_image_capi.h"
#include "capi/vision/instance_seg/instance_seg_capi.h"

int main() {
    MDStatusCode ret;
    MDModel model;
    if (ret = md_create_instance_seg_model(&model, "../../test_data/test_models/yolo11n-seg.onnx", 8); ret) {
        std::cout << ret << std::endl;
        return ret;
    }

    const std::chrono::system_clock::time_point start = std::chrono::system_clock::now();
    auto im = md_read_image("../../test_data/test_images/test_face_detection.jpg");
    MDDetectionResults result;
    if ((ret = md_instance_seg_predict(&model, &im, &result)) != 0) {
        std::cout << ret << std::endl;
        return ret;
    }
    md_draw_instance_seg_result(&im, &result,0.5, "../../test_data/msyh.ttc", 14, 0.1, 1);
    const std::chrono::duration<double> diff = std::chrono::system_clock::now() - start;
    std::cout << "duration cost: " << diff.count() << "s" << std::endl;
    md_show_image(&im);
    md_print_instance_seg_result(&result);
    md_free_instance_seg_result(&result);
    md_free_image(&im);
    md_free_instance_seg_model(&model);
    return ret;
}
