//
// Created by aichao on 2026/8/12.
//
// Sophgo(算能) TPU OBB(旋转框检测) demo
//
// 用法:
//   demo_obb_sophgo <model> <image> [conf_threshold=0.5] [loop_count=100]

#include "csrc/runtime/runtime_option.h"
#include "csrc/vision/obb/ultralytics_obb.h"
#include "csrc/vision/common/image_data.h"
#include "csrc/vision/common/visualize/visualize.h"
#include "csrc/utils/benchmark.h"

#include <cstdio>
#include <cstdlib>
#include <memory>
#include <string>
#include <vector>

using namespace modeldeploy;
using namespace modeldeploy::vision;

static bool ends_with(const std::string& s, const std::string& suffix) {
    return s.size() >= suffix.size() &&
           s.compare(s.size() - suffix.size(), suffix.size(), suffix) == 0;
}

int main(int argc, char** argv) {
    const std::string model = argc > 1 ? argv[1]
        : "../../test_data/test_models/sophgo/yolo11n-obb.bmodel";
    const std::string image = argc > 2 ? argv[2]
        : "../../test_data/test_images/test_obb1.jpg";
    const float conf_threshold = argc > 3 ? static_cast<float>(atof(argv[3])) : 0.5f;
    const int loop_count = argc > 4 ? atoi(argv[4]) : 100;
    std::vector<int> obb_size = {1024, 1024};

    RuntimeOption option;
    if (ends_with(model, ".bmodel")) {
        option.use_sophgo_backend(); option.device_id = 0;
        option.sophgo_option.bmodel_path = model;
        printf("[backend] Sophgo TPU, bmodel = %s\n", model.c_str());
    } else {
        option.use_ort_backend();
        option.set_device(modeldeploy::Device::CPU);
        option.set_cpu_thread_num(4);
        printf("[backend] ORT CPU, onnx = %s\n", model.c_str());
    }

    auto obb_model = std::make_unique<detection::UltralyticsObb>(model, option);
    if (!obb_model->is_initialized()) {
        printf("model init failed: %s\n", model.c_str());
        return 1;
    }

    const auto input_shape = obb_model->get_input_info(0).shape;
    if (input_shape.size() >= 4 && input_shape[2] > 0 && input_shape[3] > 0) {
        obb_size = {static_cast<int>(input_shape[3]), static_cast<int>(input_shape[2])};
    }
    printf("[input] model input size: %dx%d\n", obb_size[0], obb_size[1]);

    obb_model->get_preprocessor().set_size(obb_size);
    obb_model->get_postprocessor().set_conf_threshold(conf_threshold);
    const auto label_map = obb_model->get_label_map("names");

    auto img = ImageData::imread(image);
    if (img.empty()) {
        printf("failed to read image: %s\n", image.c_str());
        return 1;
    }
    printf("image: %dx%d\n", img.width(), img.height());

    std::vector<ObbResult> result;
    constexpr int warming_up_count = 5;
    for (int i = 0; i < warming_up_count; ++i) {
        obb_model->predict(img, &result);
    }

    TimerArray timers;
    for (int i = 0; i < loop_count; ++i) {
        obb_model->predict(img, &result, &timers);
    }
    timers.print_benchmark();

    printf("obb_detections=%zu @conf %.2f\n", result.size(), conf_threshold);
    for (auto& r : result) {
        printf("  label=%d score=%.4f box=[%.1f %.1f %.1f %.1f] angle=%.1f\n",
               r.label_id, r.score, r.rotated_box.xc, r.rotated_box.yc,
               r.rotated_box.width, r.rotated_box.height, r.rotated_box.angle);
    }

    try {
        auto vis_image = vis_obb(img, result, conf_threshold, "", 12, 0.3, false);
        const std::string out_name = "vis_obb_sophgo.jpg";
        vis_image.imwrite(out_name);
        printf("saved %s\n", out_name.c_str());
    } catch (...) {
        printf("vis skipped\n");
    }
    return 0;
}
