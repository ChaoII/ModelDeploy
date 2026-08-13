//
// Created by aichao on 2026/8/13.
//
// Sophgo TPU 语义分割 demo (yolo26n-sem)
//
// 用法:
//   demo_sem_sophgo <model> <image> [loop_count=100]

#include "csrc/runtime/runtime_option.h"
#include "csrc/vision/sem/ultralytics_sem.h"
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
        : "../../test_data/test_models/sophgo/yolo26n/yolo26n-sem_F16.bmodel";
    const std::string image = argc > 2 ? argv[2]
        : "../../test_data/test_images/111.jpg";
    const int loop_count = argc > 3 ? atoi(argv[3]) : 50;
    std::vector<int> sem_size = {640, 640};

    RuntimeOption option;
    if (ends_with(model, ".bmodel")) {
        option.use_sophgo_backend(0);
        option.sophgo_option.bmodel_path = model;
        printf("[backend] Sophgo TPU, bmodel = %s\n", model.c_str());
    } else {
        option.use_ort_backend();
        option.use_cpu();
        option.set_cpu_thread_num(4);
        printf("[backend] ORT CPU, onnx = %s\n", model.c_str());
    }

    auto sem_model = std::make_unique<detection::UltralyticsSem>(model, option);
    if (!sem_model->is_initialized()) {
        printf("model init failed: %s\n", model.c_str());
        return 1;
    }

    const auto input_shape = sem_model->get_input_info(0).shape;
    if (input_shape.size() >= 4 && input_shape[2] > 0 && input_shape[3] > 0) {
        sem_size = {static_cast<int>(input_shape[3]), static_cast<int>(input_shape[2])};
    }
    printf("[input] model input size: %dx%d\n", sem_size[0], sem_size[1]);

    sem_model->get_preprocessor().set_size(sem_size);
    const auto label_map = sem_model->get_label_map("names");

    auto img = ImageData::imread(image);
    if (img.empty()) {
        printf("failed to read image: %s\n", image.c_str());
        return 1;
    }
    printf("image: %dx%d\n", img.width(), img.height());

    SemSegResult result;
    constexpr int warming_up_count = 3;
    for (int i = 0; i < warming_up_count; ++i) {
        sem_model->predict(img, &result);
    }

    TimerArray timers;
    for (int i = 0; i < loop_count; ++i) {
        sem_model->predict(img, &result, &timers);
    }
    timers.print_benchmark();

    printf("sem result: shape=[%lld %lld] num_classes=%d\n",
           static_cast<long long>(result.shape[0]), static_cast<long long>(result.shape[1]),
           result.num_classes);

    try {
        auto vis_image = vis_sem(img, result, label_map, 0.5, true);
        printf("saved vis_sem.jpg\n");
    } catch (...) {
        printf("vis skipped\n");
    }
    return 0;
}
