// ModelDeploy demo: 姿态估计（trt）。
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
    opt.use_trt_backend();
    opt.set_device(modeldeploy::Device::GPU, 0);
    opt.enable_fp16 = true;

    // ---- 2. 加载模型（姿态估计）----
    auto m = std::make_unique<modeldeploy::vision::detection::UltralyticsPose>("../../test_data/test_models/trt/yolo26n/yolo26n-pose.engine", opt);
    if (!m->is_initialized()) { std::fprintf(stderr, "init failed\n"); return 1; }
    m->get_postprocessor().set_keypoints_num(17);
    // ---- 3. 读图 ----
    auto im = modeldeploy::vision::ImageData::imread("../../test_data/test_images/test_person.jpg");
    if (im.empty()) { std::fprintf(stderr, "cannot read image\n"); return 1; }

    std::vector<modeldeploy::vision::KeyPointsResult> res; // 推理结果

    // ---- 4. 推理：先 warmup，再计时 ----
    for (int i = 0; i < 20; ++i) m->predict(im, &res, nullptr);
    TimerArray timers;
    for (int i = 0; i < 80; ++i) m->predict(im, &res, &timers);
    timers.print_benchmark();

    // ---- 5. 结果与可视化 ----
    auto vis = modeldeploy::vision::vis_pose(im, res, kFont, 12, 4, 0.3, false);
    (void)vis.imwrite("result_pose_trt.jpg");
    std::printf("done, %zu persons\n", res.size());
    return 0;

}
