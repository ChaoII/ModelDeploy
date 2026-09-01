// ModelDeploy demo: 人脸检测（ort_gpu_cuda_ep）。
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

    // ---- 2. 加载模型（人脸检测）----
    auto m = std::make_unique<modeldeploy::vision::face::Scrfd>("../../test_data/test_models/onnx/seetaface/scrfd_2.5g_bnkps_shape640x640.onnx", opt);
    if (!m->is_initialized()) { std::fprintf(stderr, "init failed\n"); return 1; }
    // ---- 3. 读图 ----
    auto im = modeldeploy::vision::ImageData::imread("../../test_data/test_images/test_face_detection4.jpg");
    if (im.empty()) { std::fprintf(stderr, "cannot read image\n"); return 1; }
    auto im_bak = im.clone();

    std::vector<modeldeploy::vision::KeyPointsResult> res; // 推理结果

    // ---- 4. 推理：先 warmup，再计时 ----
    for (int i = 0; i < 10; ++i) m->predict(im, &res, nullptr);
    TimerArray timers;
    for (int i = 0; i < 50; ++i) m->predict(im, &res, &timers);
    timers.print_benchmark();

    // ---- 5. 结果与可视化 ----
    modeldeploy::vision::dis_lmk(res);
    auto vis = modeldeploy::vision::vis_keypoints(im_bak, res, kFont, 14, 2, 0.3, false, true);
    (void)vis.imwrite("result_face_det_ort_gpu_cuda_ep.jpg");
    std::printf("done, %zu faces\n", res.size());
    return 0;

}
