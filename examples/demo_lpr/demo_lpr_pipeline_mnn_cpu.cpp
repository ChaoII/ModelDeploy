// ModelDeploy demo: 车牌识别（mnn_cpu）。
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

    // ---- 2. 加载模型（车牌识别：检测 + 识别）----
    auto m = std::make_unique<modeldeploy::vision::lpr::LprPipeline>("../../test_data/test_models/onnx/yolov5plate.onnx", "../../test_data/test_models/onnx/plate_recognition_color.onnx", opt);
    if (!m->is_initialized()) { std::fprintf(stderr, "init failed\n"); return 1; }
    // ---- 3. 读图 ----
    auto im = modeldeploy::vision::ImageData::imread("../../test_data/test_images/test_lpr_detection.jpg");
    if (im.empty()) { std::fprintf(stderr, "cannot read image\n"); return 1; }
    auto im_bak = im.clone();

    std::vector<modeldeploy::vision::LprResult> res; // 推理结果

    // ---- 4. 推理：先 warmup，再计时 ----
    for (int i = 0; i < 5; ++i) m->predict(im, &res, nullptr);
    TimerArray timers;
    for (int i = 0; i < 20; ++i) m->predict(im, &res, &timers);
    timers.print_benchmark();

    // ---- 5. 结果与可视化 ----
    auto vis = modeldeploy::vision::vis_lpr(im_bak, res, kFont);
    (void)vis.imwrite("result_lpr_pipeline_mnn_cpu.jpg");
    std::printf("done, %zu plates\n", res.size());
    return 0;

}
