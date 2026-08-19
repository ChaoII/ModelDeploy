// ModelDeploy demo: 图像分类（mnn_cpu）。
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
    opt.use_cpu();

    // ---- 2. 加载模型（图像分类）----
    auto m = std::make_unique<modeldeploy::vision::classification::Classification>("../../test_data/test_models/mnn/yolo26n/yolo26n-cls.mnn", opt);
    if (!m->is_initialized()) { std::fprintf(stderr, "init failed\n"); return 1; }
    m->get_preprocessor().set_size({224, 224});
    m->get_preprocessor().disable_center_crop();
    // 自动判别单标签/多标签（yolo-cls 单标签 softmax 和≈1）
    m->get_postprocessor().set_multi_label_auto(true);
    // ---- 3. 读图 ----
    auto im = modeldeploy::vision::ImageData::imread("../../test_data/test_images/best_0.jpg");
    if (im.empty()) { std::fprintf(stderr, "cannot read image\n"); return 1; }

    // ---- 4. 推理（warmup）----
    modeldeploy::vision::ClassifyResult res;
    for (int i = 0; i < 10; ++i) m->predict(im, &res);
    for (int i = 0; i < 50; ++i) m->predict(im, &res);

    // ---- 5. 结果与可视化 ----
    modeldeploy::vision::dis_cls(res);
    auto vis = modeldeploy::vision::vis_cls(im, res, 5, 0.5, kFont, 12, 0.3, false);
    (void)vis.imwrite("result_classification_mnn_cpu.jpg");
    std::printf("done, label=%d score=%.4f\n",
                res.label_ids.empty() ? -1 : res.label_ids[0],
                res.scores.empty() ? -1.f : res.scores[0]);
    return 0;

}
