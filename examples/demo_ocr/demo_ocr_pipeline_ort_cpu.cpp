// ModelDeploy demo: 光学字符识别（ort_cpu）。
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

    // ---- 2. 加载模型（OCR：检测 + 方向分类 + 识别；词典统一 ppocrv6_tiny）----
    const char* dict = "../../test_data/test_models/onnx/dx_ocr/dx_dict.txt";
    auto m = std::make_unique<modeldeploy::vision::ocr::PaddleOCR>(
        "../../test_data/test_models/onnx/dx_ocr/det.onnx",
        "",
        "../../test_data/test_models/onnx/dx_ocr/rec.onnx", dict, opt);
    if (!m->is_initialized()) {
        std::fprintf(stderr, "init failed\n");
        return 1;
    }
    // m->set_rec_batch_size(8);
    // m->get_detector()->get_postprocessor().set_det_db_thresh(0.9);
    m->get_detector()->get_postprocessor().set_det_db_box_thresh(0.7);
    // ---- 3. 读图 ----
    auto im = modeldeploy::vision::ImageData::imread("../../test_data/test_images/sdd9.jpg");
    if (im.empty()) {
        std::fprintf(stderr, "cannot read image\n");
        return 1;
    }

    modeldeploy::vision::OCRResult res; // 推理结果

    // ---- 4. 推理：先 warmup，再计时 ----
    for (int i = 0; i < 5; ++i) m->predict(im, &res, nullptr);
    TimerArray timers;
    for (int i = 0; i < 20; ++i) m->predict(im, &res, &timers);
    timers.print_benchmark();

    // ---- 5. 结果与可视化 ----
    modeldeploy::vision::dis_ocr(res);
    auto vis = modeldeploy::vision::vis_ocr(im, res, kFont, 26);
    vis.imshow("123");
    (void)vis.imwrite("result_ocr_pipeline_ort_cpu.jpg");
    std::printf("done, %zu boxes\n", res.boxes.size());
    return 0;
}
