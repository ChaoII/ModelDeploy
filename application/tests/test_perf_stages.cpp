#include <catch2/catch_test_macros.hpp>
#include <string>
#include <vector>

#include "config.hpp"
#include "runtime_factory.hpp"
#include "csrc/serving/adapters.h"
#include "csrc/vision/common/image_data.h"
#include "csrc/utils/benchmark.h"

using namespace modeldeploy;
using namespace modeldeploy::vision;

static void run_stages(const std::string& label, const ModelConfig& cfg) {
    using RM = serving::ResultModel<detection::UltralyticsDet, std::vector<DetectionResult>>;
    RuntimeOption opt = build_runtime_option(cfg);
    RM model(cfg.path, opt);
    if (!model.is_initialized()) { SUCCEED(label + " init failed (skip)"); return; }
    if (cfg.input_size.size() == 2) model.get_preprocessor().set_size(cfg.input_size);
    ImageData img(640, 640, MdImageType::PKG_BGR_U8);

    for (int i = 0; i < 5; ++i) { std::vector<DetectionResult> r; model.predict(img, &r, nullptr); }
    TimerArray t;
    for (int i = 0; i < 30; ++i) { std::vector<DetectionResult> r; model.predict(img, &r, &t); }
    INFO(label << " single: pre=" << t.pre_timer.mean_ms() << "ms infer=" << t.infer_timer.mean_ms()
               << "ms post=" << t.post_timer.mean_ms() << "ms total=" << t.mean_ms() << "ms");

    std::vector<ImageData> imgs(8, img);
    TimerArray bt;
    for (int i = 0; i < 10; ++i) {
        std::vector<std::vector<DetectionResult>> r;
        model.batch_predict(imgs, &r, &bt);
    }
    INFO(label << " batch8: pre=" << bt.pre_timer.mean_ms() << "ms infer=" << bt.infer_timer.mean_ms()
               << "ms post=" << bt.post_timer.mean_ms() << "ms total=" << bt.mean_ms()
               << "ms (/8=" << bt.mean_ms() / 8.0 << "ms)");
    SUCCEED("stages measured: " + label);
}

// 分阶段耗时剖析：pre/infer/post（单帧 + batch8），多运行时对比。
TEST_CASE("perf stages yolo11n", "[perf]") {
    ModelConfig base;
    base.name = "yolo11n";
    base.type = "detection";
    base.path = "test_data/test_models/onnx/yolo11n/yolo11n.onnx";
    base.input_size = {640, 640};
    base.confidence_threshold = 0.25f;
    base.backend = "ort";

    { auto c = base; c.device = "cpu"; c.use_trt_ep = false; run_stages("cpu", c); }
    { auto c = base; c.device = "gpu"; c.use_trt_ep = false; run_stages("gpu-cudaep", c); }
    { auto c = base; c.device = "gpu"; c.use_trt_ep = true;  run_stages("gpu-trt", c); }
}
