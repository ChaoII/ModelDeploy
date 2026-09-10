// ModelDeploy demo: 行人属性（ort_cpu）。
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
    opt.set_device(modeldeploy::Device::CPU);
    opt.set_cpu_thread_num(4);

    // ---- 2. 加载模型（行人属性：检测 + 多标签分类）----
    auto m = std::make_unique<modeldeploy::vision::pipeline::PedestrianAttribute>(
        "../../best.onnx", "../../zhgd_ml_c.onnx", opt);
    if (!m->is_initialized()) {
        std::fprintf(stderr, "init failed\n");
        return 1;
    }
    m->set_cls_batch_size(8);
    m->set_det_input_size({1280, 1280});
    m->set_det_threshold(0.5);
    m->set_cls_input_size({192, 256});
    // ---- 3. 读图 ----
    auto im = modeldeploy::vision::ImageData::imread("../../test_data/test_images/test_pedestrian_attribute2.jpg");
    if (im.empty()) {
        std::fprintf(stderr, "cannot read image\n");
        return 1;
    }
    std::vector<modeldeploy::vision::AttributeResult> res; // 推理结果

    // ---- 4. 推理：先 warmup，再计时 ----
    for (int i = 0; i < 10; ++i) m->predict(im, &res, nullptr);
    TimerArray timers;
    for (int i = 0; i < 50; ++i) m->predict(im, &res, &timers);
    timers.print_benchmark();

    // ---- 5. 结果与可视化 ----
    modeldeploy::vision::dis_attr(res);
    std::unordered_map<int, std::string> label_map;
    label_map[0] = "安全帽";
    label_map[1] = "反光衣";
    label_map[2] = "安全绳";
    label_map[3] = "工作服";
    label_map[4] = "安全装置";
    std::vector<int> abnormal;
    for (int i = 0; i < res.size(); i++) {
        if (res[i].attr_scores[4] < 0.5) {
            abnormal.push_back(i);
        }
    }
    auto vis = modeldeploy::vision::vis_attr(im, res, 0.5, label_map, kFont, 15, 0.15, false, abnormal);
    vis.imshow("ss");
    (void)vis.imwrite("result_pedestrian_attribute_ort_cpu2.jpg");
    std::printf("done, %zu persons\n", res.size());
    return 0;
}
