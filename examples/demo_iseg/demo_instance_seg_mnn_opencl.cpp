// ModelDeploy demo: 实例分割（mnn_opencl）。
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
    opt.set_device(modeldeploy::Device::OPENCL, 0);

    // ---- 2. 加载模型（实例分割）----
    auto m = std::make_unique<modeldeploy::vision::detection::UltralyticsSeg>("../../test_data/test_models/mnn/yolo26n/yolo26n-seg.mnn", opt);
    if (!m->is_initialized()) { std::fprintf(stderr, "init failed\n"); return 1; }
    // ---- 3. 读图 ----
    auto im = modeldeploy::vision::ImageData::imread("../../test_data/test_images/test_person.jpg");
    if (im.empty()) { std::fprintf(stderr, "cannot read image\n"); return 1; }

    std::vector<modeldeploy::vision::InstanceSegResult> res; // 推理结果

    // ---- 4. 推理：先 warmup，再计时 ----
    for (int i = 0; i < 10; ++i) m->predict(im, &res, nullptr);
    TimerArray timers;
    for (int i = 0; i < 100; ++i) m->predict(im, &res, &timers);
    timers.print_benchmark();

    // ---- 5. 结果与可视化 ----
    auto vis = modeldeploy::vision::vis_iseg(im, res, 0.2, kFont, 14, 0.5, false);
    (void)vis.imwrite("result_instance_seg_mnn_opencl.jpg");
    std::printf("done, %zu masks\n", res.size());
    return 0;

}
