//
// Created by aichao on 2025/2/24.
//

#include <iostream>
#include <chrono>

#include "capi/utils/md_image_capi.h"
#include "capi/vision/obb/obb_capi.h"

int main() {
    MDStatusCode ret;
    MDModel model;
    if (ret = md_create_obb_model(&model, "../../test_data/test_models/yolo11n-obb.onnx", 8); ret) {
        std::cout << ret << std::endl;
        return ret;
    }
    if (constexpr MDSize size = {1024, 1024}; (ret = md_set_obb_input_size(&model, size)) != 0) {
        std::cout << ret << std::endl;
        return ret;
    }
    const std::chrono::system_clock::time_point start = std::chrono::system_clock::now();
    auto im = md_read_image("../../test_data/test_images/test_obb1.jpg");
    MDObbResults results;
    if ((ret = md_obb_predict(&model, &im, &results)) != 0) {
        std::cout << ret << std::endl;
        return ret;
    }
    md_draw_obb_result(&im, &results, 0.5, "../../test_data/msyh.ttc", 14, 0.5, 1);
    const std::chrono::duration<double> diff = std::chrono::system_clock::now() - start;
    std::cout << "duration cost: " << diff.count() << "s" << std::endl;
    md_show_image(&im);
    md_print_obb_result(&results);
    md_free_obb_result(&results);
    md_free_image(&im);
    md_free_obb_model(&model);
    return ret;
}
