// ModelDeploy demo: 语义分割（ncnn_vulkan）。
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
    opt.use_ncnn_backend();
    opt.set_device(modeldeploy::Device::VULKAN);

    // ---- 2. 加载模型（语义分割）----
    auto m = std::make_unique<modeldeploy::vision::detection::UltralyticsSem>("../../test_data/test_models/ncnn/yolo26n/yolo26n-sem.param", opt);
    if (!m->is_initialized()) { std::fprintf(stderr, "init failed\n"); return 1; }
    const auto label_map = m->get_label_map("names");
    // ---- 3. 读图 ----
    auto im = modeldeploy::vision::ImageData::imread("../../test_data/test_images/test_sem_540.jpg");
    if (im.empty()) { std::fprintf(stderr, "cannot read image\n"); return 1; }

    modeldeploy::vision::SemSegResult res; // 推理结果

    // ---- 4. 推理：先 warmup，再计时 ----
    for (int i = 0; i < 20; ++i) m->predict(im, &res, nullptr);
    TimerArray timers;
    for (int i = 0; i < 100; ++i) m->predict(im, &res, &timers);
    timers.print_benchmark();

    // ---- 5. 结果与可视化 ----
    auto vis = modeldeploy::vision::vis_sem(im, res, label_map, 0.5, true);
    (void)vis.imwrite("result_sem_ncnn_vulkan.jpg");
    std::printf("done %zux%zu\n",
                res.shape.empty() ? 0 : (size_t)res.shape[0],
                res.shape.size() < 2 ? 0 : (size_t)res.shape[1]);
    return 0;

}
