//
// Created by aichao on 2026/8/12.
//
// Sophgo(算能) TPU 分类 demo — 支持单标签(yolo11n-cls)与多标签(zhgd_ml)
//
// 用法:
//   demo_classification_sophgo <model> <image> [topk=1] [multi_label=0] [loop_count=100]
//
// model 以 .bmodel 结尾 → Sophgo TPU 后端；否则按 ONNX 走 ORT CPU(便于对照)。

#include "csrc/runtime/runtime_option.h"
#include "csrc/vision/classification/classification.h"
#include "csrc/vision/common/image_data.h"
#include "csrc/vision/common/visualize/visualize.h"
#include "csrc/utils/benchmark.h"

#include <cstdio>
#include <cstdlib>
#include <chrono>
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
        : "../../test_data/test_models/sophgo/yolo11n-cls.bmodel";
    const std::string image = argc > 2 ? argv[2]
        : "../../test_data/test_images/test_face.jpg";
    const int topk = argc > 3 ? atoi(argv[3]) : 1;
    const bool multi_label = argc > 4 ? atoi(argv[4]) != 0 : false;
    const int loop_count = argc > 5 ? atoi(argv[5]) : 100;
    std::vector<int> cls_size = {224, 224};

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

    auto cls_model = std::make_unique<classification::Classification>(model, option);
    if (!cls_model->is_initialized()) {
        printf("model init failed: %s\n", model.c_str());
        return 1;
    }

    // 输入尺寸从模型元数据获取
    const auto input_shape = cls_model->get_input_info(0).shape;
    if (input_shape.size() >= 4 && input_shape[2] > 0 && input_shape[3] > 0) {
        cls_size = {static_cast<int>(input_shape[3]), static_cast<int>(input_shape[2])};
    }
    printf("[input] model input size: %dx%d\n", cls_size[0], cls_size[1]);

    cls_model->get_preprocessor().set_size(cls_size);
    cls_model->get_preprocessor().disable_center_crop();
    cls_model->get_postprocessor().set_top_k(topk);
    cls_model->get_postprocessor().set_multi_label(multi_label);

    auto img = ImageData::imread(image);
    if (img.empty()) {
        printf("failed to read image: %s\n", image.c_str());
        return 1;
    }
    printf("image: %dx%d\n", img.width(), img.height());

    ClassifyResult result;
    constexpr int warming_up_count = 5;
    for (int i = 0; i < warming_up_count; ++i) {
        cls_model->predict(img, &result);
    }

    const auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < loop_count; ++i) {
        cls_model->predict(img, &result);
    }
    const auto t1 = std::chrono::steady_clock::now();
    const double avg_ms = std::chrono::duration<double, std::milli>(t1 - t0).count() / loop_count;
    printf("[predict] avg = %.2f ms (loop=%d)\n", avg_ms, loop_count);

    printf("topk=%d multi_label=%d\n", topk, multi_label ? 1 : 0);
    for (size_t i = 0; i < result.label_ids.size(); ++i) {
        printf("  label=%d score=%.4f\n", result.label_ids[i], result.scores[i]);
    }

    try {
        auto vis_image = vis_cls(img, result, topk, 0.5f, "", 12, 0.3, false);
        const std::string out_name = "vis_cls_sophgo.jpg";
        vis_image.imwrite(out_name);
        printf("saved %s\n", out_name.c_str());
    } catch (...) {
        printf("vis skipped\n");
    }
    return 0;
}
