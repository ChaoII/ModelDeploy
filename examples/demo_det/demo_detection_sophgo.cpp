//
// Created by aichao on 2026/8/3.
//
// Sophgo(算能) TPU 检测 demo — 与 demo_detection_cxx.cpp 同风格的 TPU 版
//
// 用法:
//   demo_detection_sophgo <model> <image> [conf_threshold=0.5] [loop_count=100] [font_path]
//
// model 以 .bmodel 结尾 → Sophgo TPU 后端(需 ENABLE_SOPHGO 编译)；
// 否则按 ONNX 走 ORT CPU(便于本机无 TPU 时对照)。
//
// 注意: YOLO 模型 (yolo11n_without_nms 等) 期望 [0,1] 归一化输入，SDK 默认
// letterbox + /255 预处理即正确约定，无需 set_normalize(false)。

#include "csrc/runtime/runtime_option.h"
#include "csrc/vision/detection/ultralytics_det.h"
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
    const std::string model = argc > 1 ? argv[1] : "yolo11n_bm1688.bmodel";
    const std::string image = argc > 2 ? argv[2] : "test.jpg";
    const float conf_threshold = argc > 3 ? static_cast<float>(atof(argv[3])) : 0.5f;
    const int loop_count = argc > 4 ? atoi(argv[4]) : 100;
    const std::string font_path = argc > 5 ? argv[5] : "";
    std::vector<int> det_size = {640, 640};

    RuntimeOption option;
    if (ends_with(model, ".bmodel")) {
        option.use_sophgo_backend(0);
        option.sophgo_option.bmodel_path = model;
        printf("[backend] Sophgo TPU, bmodel = %s\n", model.c_str());
        // bmodel 输入尺寸固定(转换时 --input_shapes 指定)，需与预处理输出一致
        det_size = {1280, 1280};
    } else {
        option.use_ort_backend();
        option.use_cpu();
        option.set_cpu_thread_num(4);
        printf("[backend] ORT CPU, onnx = %s\n", model.c_str());
    }

    auto det = std::make_unique<detection::UltralyticsDet>(model, option);
    if (!det->is_initialized()) {
        printf("model init failed: %s\n", model.c_str());
        return 1;
    }

    // SDK 默认预处理即 letterbox + /255 归一化到 [0,1]，符合 Ultralytics 训练约定。
    // 无 NMS 模型建议阈值取 0.5 以上(0.25 会带出大量低分候选)。
    det->get_preprocessor().set_size(det_size);
    det->get_postprocessor().set_conf_threshold(conf_threshold);
    const auto label_map = det->get_label_map("names");

    auto img = ImageData::imread(image);
    if (img.empty()) {
        printf("failed to read image: %s\n", image.c_str());
        return 1;
    }
    printf("image: %dx%d\n", img.width(), img.height());

    std::vector<DetectionResult> result;
    constexpr int warming_up_count = 5;
    for (int i = 0; i < warming_up_count; ++i) {
        det->predict(img, &result);
    }

    TimerArray timers;
    for (int i = 0; i < loop_count; ++i) {
        det->predict(img, &result, &timers);
    }
    timers.print_benchmark();

    const float top_score = result.empty() ? 0.0f : result[0].score;
    printf("detections=%zu @conf %.2f, top_score %.4f\n",
           result.size(), conf_threshold, top_score);
    for (auto& r : result) {
        printf("  label=%d score=%.4f box=[%.0f %.0f %.0f %.0f]\n",
               r.label_id, r.score, r.box.x, r.box.y, r.box.width, r.box.height);
    }

    // 可视化保存（OpenCV FontFace 在部分环境下偶发崩溃，失败不影响检测结果）
    try {
        auto vis_image = vis_det(img, result, conf_threshold, label_map, font_path, 12, 0.3, false);
        const std::string out_name = "vis_sophgo.jpg";
        vis_image.imwrite(out_name);
        printf("saved %s\n", out_name.c_str());
    } catch (...) {
        printf("vis skipped (font/render issue)\n");
    }
    return 0;
}
