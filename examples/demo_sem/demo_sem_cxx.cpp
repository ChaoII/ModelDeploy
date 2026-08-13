//
// Created by aichao on 2026/8/13.
//
// yolo26n-sem 本地推理 demo（onnx -> ORT/GPU/TRT）
//
// 用法:
//   demo_sem_cxx [model] [image]

#include <iostream>

#include "csrc/vision.h"
#include "csrc/vision/common/visualize/visualize.h"


int main(int argc, char** argv) {
    modeldeploy::RuntimeOption option;
    option.use_gpu();
    option.use_ort_backend();
    option.enable_fp16 = true;
    const std::string model_file = argc > 1 ? argv[1]
        : "../../test_data/test_models/onnx/yolo26n/yolo26n-sem.onnx";
    const std::string image_file = argc > 2 ? argv[2]
        : "../../test_data/test_images/111.jpg";
    auto model = modeldeploy::vision::detection::UltralyticsSem(model_file, option);
    model.get_preprocessor().use_cuda_preproc();
    auto im = modeldeploy::vision::ImageData::imread(image_file);
    modeldeploy::vision::SemSegResult res;
    constexpr int loop = 20;
    TimerArray times;
    for (int i = 0; i < loop; ++i) {
        model.predict(im, &res, &times);
    }
    times.print_benchmark();
    std::cout << "sem result: shape=[" << res.shape[0] << " " << res.shape[1]
              << "] num_classes=" << res.num_classes << std::endl;
    const auto label_map = model.get_label_map("names");
    auto vis_im = modeldeploy::vision::vis_sem(im, res, label_map, 0.5, true);
    std::cout << "saved vis_sem.jpg" << std::endl;
    return 0;
}
