//
// Created by aichao on 2026/8/18.
//
// Sophgo(算能) TPU 行人属性 demo — demo_pedestrian_attribute_cxx.cpp 的 TPU 版
//
// 用法:
//   demo_pedestrian_attribute_sophgo <det_model> <ml_model> <image>
//                                     [loop_count=100] [conf_threshold=0.5] [font_path]
//
// det/ml 以 .bmodel 结尾 → Sophgo TPU 后端(需要 ENABLE_SOPHGO 编译)；
// 否则按 ONNX 走 ORT CPU(便于无 TPU 对照)。
//
// 关键点: Pipeline 内检测/分类两个子模型共享同一个 RuntimeOption，
//        但每个子模型构造会用 set_model_path 写入自己的 model_file；
//        Sophgo 后端在 bmodel_path 为空时回退到 option.model_file，
//        因此这里不设 bmodel_path，两个 bmodel 各自从构造参数加载。

#include "csrc/vision.h"
#include "csrc/vision/common/display/display.h"
#include "csrc/vision/common/visualize/visualize.h"

#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

using namespace modeldeploy;
using namespace modeldeploy::vision;

static bool ends_with(const std::string& s, const std::string& suffix) {
    return s.size() >= suffix.size() &&
           s.compare(s.size() - suffix.size(), suffix.size(), suffix) == 0;
}

int main(int argc, char** argv) {
    const std::string det_model = argc > 1 ? argv[1]
        : "../../test_data/test_models/sophgo/zhgd_without_nms_640_int8.bmodel";
    const std::string ml_model = argc > 2 ? argv[2]
        : "../../test_data/test_models/sophgo/zhgd_ml_int8.bmodel";
    const std::string image_path = argc > 3 ? argv[3]
        : "../../test_data/test_images/test_pedestrian_attribute_scale.png";
    const int loop_count = argc > 4 ? atoi(argv[4]) : 100;
    const float conf_threshold = argc > 5 ? static_cast<float>(atof(argv[5])) : 0.5f;
    const std::string font_path = argc > 6 ? argv[6] : "";

    RuntimeOption option;
    if (ends_with(det_model, ".bmodel")) {
        option.use_sophgo_backend(); option.device_id = 0;          // 不设 bmodel_path → 各子模型用自己的 model_file
        printf("[backend] Sophgo TPU: det=%s\n", det_model.c_str());
    } else {
        option.use_ort_backend();
        option.use_cpu();
        option.set_cpu_thread_num(4);
        printf("[backend] ORT CPU: det=%s\n", det_model.c_str());
    }

    pipeline::PedestrianAttribute pedestrian_attribute(det_model, ml_model, option);
    if (!pedestrian_attribute.is_initialized()) {
        printf("pedestrian attribute model init failed\n");
        return 1;
    }

    std::vector<int> det_size = {640, 640};
    const auto det_shape = pedestrian_attribute.get_detector()->get_input_info(0).shape;
    if (det_shape.size() >= 4 && det_shape[2] > 0 && det_shape[3] > 0) {
        det_size = {static_cast<int>(det_shape[3]), static_cast<int>(det_shape[2])};
    }
    printf("[input] det %dx%d\n", det_size[0], det_size[1]);

    pedestrian_attribute.set_det_input_size(det_size);
    pedestrian_attribute.set_det_threshold(conf_threshold);
    // 分类输入同样从 bmodel 元数据读取（zhgd_ml 的实际输入，勿硬编码）
    std::vector<int> cls_size = {192, 256};
    const auto cls_shape = pedestrian_attribute.get_classifier()->get_input_info(0).shape;
    if (cls_shape.size() >= 4 && cls_shape[2] > 0 && cls_shape[3] > 0) {
        cls_size = {static_cast<int>(cls_shape[3]), static_cast<int>(cls_shape[2])};
    }
    printf("[input] cls %dx%d\n", cls_size[0], cls_size[1]);
    pedestrian_attribute.set_cls_input_size(cls_size);
    pedestrian_attribute.set_cls_batch_size(1);   // Sophgo 的 int8 bmodel 是 batch=1 静态形状，禁用动态 batch

    auto img = ImageData::imread(image_path);
    if (img.empty()) {
        printf("failed to read image: %s\n", image_path.c_str());
        return 1;
    }
    printf("image: %dx%d\n", img.width(), img.height());

    std::vector<AttributeResult> results;
    constexpr int warming_up_count = 5;
    for (int i = 0; i < warming_up_count; ++i) {
        pedestrian_attribute.predict(img, &results);
    }
    TimerArray timers;
    for (int i = 0; i < loop_count; ++i) {
        pedestrian_attribute.predict(img, &results, &timers);
    }
    timers.print_benchmark();

    printf("attributes=%zu\n", results.size());
    dis_attr(results);

    std::unordered_map<int, std::string> label_map;
    label_map.insert({0, "safety_helmet"});
    label_map.insert({1, "reflective_vest"});
    label_map.insert({2, "safety_rope"});
    label_map.insert({3, "work_uniform"});
    try {
        const auto vis_image =
            vis_attr(img, results, conf_threshold, label_map, font_path, 6, 0.15, true, {0, 1});
        (void)vis_image.imwrite("pedestrian_attr_sophgo_out.jpg");
        printf("saved pedestrian_attr_sophgo_out.jpg\n");
    } catch (...) {
        printf("vis skipped (font/render issue)\n");
    }
    return 0;
}
