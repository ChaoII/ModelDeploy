//
// Created by aichao on 2026/8/13.
//
// yolo26n-depth 本地推理 demo（onnx -> ORT/GPU/TRT）
//
// 用法:
//   demo_depth_cxx [model] [image]

#include <iostream>

#include "csrc/vision.h"
#include "csrc/vision/common/visualize/visualize.h"


int main(int argc, char** argv) {
    modeldeploy::RuntimeOption option;
    option.use_ort_backend();
    const std::string model_file = argc > 1
                                       ? argv[1]
                                       : "../../test_data/test_models/onnx/yolo26n/yolo26n-depth.onnx";
    const std::string image_file = argc > 2
                                       ? argv[2]
                                       : "../../test_data/test_images/test_depth_540.jpg";
    auto model = modeldeploy::vision::detection::UltralyticsDepth(model_file, option);
    auto im = modeldeploy::vision::ImageData::imread(image_file);
    modeldeploy::vision::DepthResult res;
    constexpr int warming_up_count = 20;
    for (int i = 0; i < warming_up_count; ++i) {
        model.predict(im, &res);
    }
    constexpr int loop_count = 100;
    TimerArray times;
    for (int i = 0; i < loop_count; ++i) {
        model.predict(im, &res, &times);
    }
    times.print_benchmark();
    std::cout << "depth result: shape=[" << res.shape[0] << " " << res.shape[1] << "]" << std::endl;
    auto vis_im = modeldeploy::vision::vis_depth(im, res, true, true);
    (void)vis_im.imwrite("depth_out.jpg");
    std::cout << "saved depth_out.jpg" << std::endl;
    return 0;
}
