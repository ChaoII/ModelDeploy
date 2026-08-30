// ModelDeploy demo: 轻量分割一切（FastSAM）。
// 读输入图 → FastSam::predict → vis_iseg 可视化保存。
#include "csrc/vision.h"
#include "csrc/vision/common/visualize/visualize.h"

#include <cstdio>
#include <memory>
#include <string>
#include <vector>

int main() {
    const char* kFont = "../../test_data/msyh.ttc";
    const char* model = "../../test_data/test_models/onnx/fastsam/fastsam-s.onnx";

    modeldeploy::RuntimeOption opt;
    opt.use_ort_backend();
    opt.use_cpu();
    opt.set_cpu_thread_num(4);

    auto m = std::make_unique<modeldeploy::vision::seg::FastSam>(model, opt);
    if (!m->is_initialized()) {
        std::fprintf(stderr, "FastSAM init failed: %s\n", model);
        return 1;
    }
    auto im = modeldeploy::vision::ImageData::imread("../../test_data/test_images/test_detection0.jpg");
    if (im.empty()) {
        std::fprintf(stderr, "cannot read image\n");
        return 1;
    }

    std::vector<modeldeploy::vision::InstanceSegResult> res;
    for (int i = 0; i < 5; ++i) m->predict(im, &res, nullptr);

    auto vis = modeldeploy::vision::vis_iseg(im, res, 0.3, kFont, 14, 0.5, false);
    (void)vis.imwrite("result_fastsam.jpg");
    std::printf("done, %zu masks\n", res.size());
    return 0;
}
