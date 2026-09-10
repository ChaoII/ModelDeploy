// ModelDeploy demo: 深度估计（mnn_cpu）。
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
    opt.use_mnn_backend();
    opt.set_device(modeldeploy::Device::CPU);

    // ---- 2. 加载模型（深度估计）----
    auto m = std::make_unique<modeldeploy::vision::detection::UltralyticsDepth>("../../test_data/test_models/mnn/yolo26n/yolo26n-depth.mnn", opt);
    if (!m->is_initialized()) { std::fprintf(stderr, "init failed\n"); return 1; }
    // ---- 3. 读图 ----
    auto im = modeldeploy::vision::ImageData::imread("../../test_data/test_images/test_depth_540.jpg");
    if (im.empty()) { std::fprintf(stderr, "cannot read image\n"); return 1; }

    modeldeploy::vision::DepthResult res; // 推理结果

    // ---- 4. 推理：先 warmup，再计时 ----
    for (int i = 0; i < 20; ++i) m->predict(im, &res, nullptr);
    TimerArray timers;
    for (int i = 0; i < 100; ++i) m->predict(im, &res, &timers);
    timers.print_benchmark();

    // ---- 5. 结果与可视化 ----
    auto vis = modeldeploy::vision::vis_depth(im, res, true, false);
    (void)vis.imwrite("result_depth_mnn_cpu.jpg");
    std::printf("done %zux%zu\n",
                res.shape.empty() ? 0 : (size_t)res.shape[0],
                res.shape.size() < 2 ? 0 : (size_t)res.shape[1]);
    return 0;

}
