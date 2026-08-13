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
    option.use_gpu();
    option.use_ort_backend();
    option.enable_fp16 = true;
    const std::string model_file = argc > 1 ? argv[1]
        : "../../test_data/test_models/onnx/yolo26n/yolo26n-depth.onnx";
    const std::string image_file = argc > 2 ? argv[2]
        : "../../test_data/test_images/111.jpg";
    auto model = modeldeploy::vision::detection::UltralyticsDepth(model_file, option);
    model.get_preprocessor().use_cuda_preproc();
    auto im = modeldeploy::vision::ImageData::imread(image_file);
    modeldeploy::vision::DepthResult res;
    constexpr int loop = 20;
    TimerArray times;
    for (int i = 0; i < loop; ++i) {
        model.predict(im, &res, &times);
    }
    times.print_benchmark();
    std::cout << "depth result: shape=[" << res.shape[0] << " " << res.shape[1] << "]" << std::endl;
    auto vis_im = modeldeploy::vision::vis_depth(im, res, true, true);
    std::cout << "saved vis_depth.jpg" << std::endl;
    return 0;
}
