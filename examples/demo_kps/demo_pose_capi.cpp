//
// Created by aichao on 2025/2/24.
//

#include <iostream>
#include <chrono>

#include "capi/utils/md_image_capi.h"
#include "capi/utils/md_utils_capi.h"
#include "capi/vision/pose/pose_capi.h"

int main() {
    MDStatusCode ret;
    MDModel model;
    const MDRuntimeOption option = md_create_default_runtime_option();
    if (ret = md_create_keypoint_model(&model, "../../test_data/test_models/zc.onnx", &option); ret) {
        std::cout << ret << std::endl;
        return ret;
    }
    if (constexpr MDSize size = {640, 640}; (ret = md_set_keypoint_input_size(&model, size)) != 0) {
        std::cout << ret << std::endl;
        return ret;
    }
    md_set_keypoint_num(&model, 2);
    const std::chrono::system_clock::time_point start = std::chrono::system_clock::now();
    auto im = md_read_image("../../test_data/test_images/zc0.jpg");
    MDKeyPointResults result;
    if ((ret = md_keypoint_predict(&model, &im, &result)) != 0) {
        std::cout << ret << std::endl;
        return ret;
    }
    md_draw_keypoint_result(&im, &result, "../../test_data/msyh.ttc", 14, 4, 0.5, 0);
    const std::chrono::duration<double> diff = std::chrono::system_clock::now() - start;
    std::cout << "duration cost: " << diff.count() << "s" << std::endl;
    md_show_image(&im);
    md_print_keypoint_result(&result);
    md_free_keypoint_result(&result);
    md_free_image(&im);
    md_free_keypoint_model(&model);
    return ret;
}
