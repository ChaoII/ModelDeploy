//
// Created by aichao on 2026/8/13.
//
// Sophgo TPU 深度估计 demo (yolo26n-depth)
//
// 用法:
//   demo_depth_sophgo <model> <image> [loop_count=100]

#include "csrc/runtime/runtime_option.h"
#include "csrc/vision/depth/ultralytics_depth.h"
#include "csrc/vision/common/image_data.h"
#include "csrc/vision/common/visualize/visualize.h"
#include "csrc/utils/benchmark.h"

#include <cstdio>
#include <cstdlib>
#include <memory>
#include <string>

using namespace modeldeploy;
using namespace modeldeploy::vision;

static bool ends_with(const std::string& s, const std::string& suffix) {
    return s.size() >= suffix.size() &&
           s.compare(s.size() - suffix.size(), suffix.size(), suffix) == 0;
}

int main(int argc, char** argv) {
    const std::string model = argc > 1 ? argv[1]
        : "../../test_data/test_models/sophgo/yolo26n/yolo26n-depth-f16.bmodel";
    const std::string image = argc > 2 ? argv[2]
        : "../../test_data/test_images/test_depth_540.jpg";
    const int loop_count = argc > 3 ? atoi(argv[3]) : 50;
    std::vector<int> depth_size = {640, 640};

    RuntimeOption option;
    if (ends_with(model, ".bmodel")) {
        option.use_sophgo_backend(); option.device_id = 0;
        option.sophgo_option.bmodel_path = model;
        printf("[backend] Sophgo TPU, bmodel = %s\n", model.c_str());
    } else {
        option.use_ort_backend();
        option.use_cpu();
        option.set_cpu_thread_num(4);
        printf("[backend] ORT CPU, onnx = %s\n", model.c_str());
    }

    auto depth_model = std::make_unique<detection::UltralyticsDepth>(model, option);
    if (!depth_model->is_initialized()) {
        printf("model init failed: %s\n", model.c_str());
        return 1;
    }

    const auto input_shape = depth_model->get_input_info(0).shape;
    if (input_shape.size() >= 4 && input_shape[2] > 0 && input_shape[3] > 0) {
        depth_size = {static_cast<int>(input_shape[3]), static_cast<int>(input_shape[2])};
    }
    printf("[input] model input size: %dx%d\n", depth_size[0], depth_size[1]);

    depth_model->get_preprocessor().set_size(depth_size);

    auto img = ImageData::imread(image);
    if (img.empty()) {
        printf("failed to read image: %s\n", image.c_str());
        return 1;
    }
    printf("image: %dx%d\n", img.width(), img.height());

    DepthResult result;
    constexpr int warming_up_count = 3;
    for (int i = 0; i < warming_up_count; ++i) {
        depth_model->predict(img, &result);
    }

    TimerArray timers;
    for (int i = 0; i < loop_count; ++i) {
        depth_model->predict(img, &result, &timers);
    }
    timers.print_benchmark();

    printf("depth result: shape=[%lld %lld]\n",
           static_cast<long long>(result.shape[0]), static_cast<long long>(result.shape[1]));

    try {
        auto vis_image = vis_depth(img, result, true, true);
        printf("saved vis_depth.jpg\n");
    } catch (...) {
        printf("vis skipped\n");
    }
    return 0;
}
