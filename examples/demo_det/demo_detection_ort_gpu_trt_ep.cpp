// ModelDeploy demo: 目标检测（ort_gpu_trt_ep）。
// 最小可运行示例，完整逻辑自包含：构造 RuntimeOption -> 加载模型 -> 预处理 -> 推理(计时) -> 可视化。
#include "csrc/vision.h"
#include "csrc/vision/common/display/display.h"
#include "csrc/vision/common/visualize/visualize.h"
#include "csrc/utils/benchmark.h"

#include <cstdio>
#include <memory>
#include <string>
#include <vector>

int main() {
    const char* kFont = "../../test_data/msyh.ttc";

    // ---- 1. 运行时选项 ----

    modeldeploy::RuntimeOption opt;
    opt.use_ort_backend();
    opt.use_gpu(0);
    opt.enable_trt = true;
    opt.enable_fp16 = true;
    opt.ort_option.trt_engine_cache_path = "./trt_engine";
    opt.set_trt_min_shape("x:1x3x640x640");
    opt.set_trt_opt_shape("x:1x3x640x640");
    opt.set_trt_max_shape("x:1x3x640x640");

    // ---- 2. 加载模型（目标检测）----
    auto det = std::make_unique<modeldeploy::vision::detection::UltralyticsDet>("../../test_data/test_models/onnx/yolo26n/yolo26n.onnx", opt);
    if (!det->is_initialized()) { std::fprintf(stderr, "init failed\n"); return 1; }
    det->get_preprocessor().set_size({640, 640});
    const auto label_map = det->get_label_map("names");
    // ---- 3. 读图 ----
    auto im = modeldeploy::vision::ImageData::imread("../../test_data/test_images/test_pedestrian_attribute_scale.png");
    if (im.empty()) { std::fprintf(stderr, "cannot read image\n"); return 1; }

    std::vector<modeldeploy::vision::DetectionResult> res; // 推理结果

    // ---- 4. 推理：先 warmup，再计时 ----
    for (int i = 0; i < 20; ++i) det->predict(im, &res, nullptr);
    TimerArray timers;
    for (int i = 0; i < 100; ++i) det->predict(im, &res, &timers);
    timers.print_benchmark();

    // ---- 5. 结果与可视化 ----
    modeldeploy::vision::dis_det(res);
    auto vis = modeldeploy::vision::vis_det(im, res, 0.5, label_map, kFont, 12, 0.3, false);
    (void)vis.imwrite("result_detection_ort_gpu_trt_ep.jpg");
    std::printf("done, %zu objects\n", res.size());
    return 0;

}
