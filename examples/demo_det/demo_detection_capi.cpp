//
// Created by aichao on 2025/2/24.
//

#include <iostream>
#include <chrono>
#include <capi/utils/md_utils_capi.h>
#include "capi/utils/md_image_capi.h"
#include "capi/vision/detection/detection_capi.h"

int main() {
    MDStatusCode ret;
    MDModel model;
    MDRuntimeOption runtime_option = md_create_default_runtime_option();
    runtime_option.cpu_thread_num = 8;
    runtime_option.device = MD_DEVICE_GPU;
    runtime_option.trt_engine_cache_path = "./trt_engine";
    runtime_option.device_id = 0;
    runtime_option.enable_fp16 = 1;
    runtime_option.enable_trt = 0;
    if (ret = md_create_detection_model(&model, "../../test_data/test_models/yolo11n_nms.onnx", &runtime_option); ret) {
        std::cout << ret << std::endl;
        return ret;
    }
    if (constexpr MDSize size = {640, 640}; (ret = md_set_detection_input_size(&model, size)) != 0) {
        std::cout << ret << std::endl;
        return ret;
    }
    auto im = md_read_image("../../test_data/test_images/test_detection0.jpg");

    const std::chrono::system_clock::time_point start = std::chrono::system_clock::now();
    MDDetectionResults result;
    if ((ret = md_detection_predict(&model, &im, &result)) != 0) {
        std::cout << ret << std::endl;
        return ret;
    }
    const std::chrono::duration<double> diff = std::chrono::system_clock::now() - start;
    md_draw_detection_result(&im, &result, 0.3, "../../test_data/msyh.ttc", 14, 0.5, 1);
    std::cout << "duration cost: " << diff.count() << "s" << std::endl;
    md_show_image(&im);
    md_print_detection_result(&result);
    md_free_detection_result(&result);
    md_free_image(&im);
    md_free_detection_model(&model);
    return ret;
}
