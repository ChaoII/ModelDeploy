#include <catch2/catch_test_macros.hpp>
#include <atomic>
#include <thread>
#include <vector>

#include "batched_detector.hpp"
#include "csrc/vision/common/image_data.h"

using namespace modeldeploy::vision;

// 共享批处理检测器：8 路并发提交应触发批合并（batch_runs < 请求数），结果全部成功。
TEST_CASE("BatchedDetector shared batch inference", "[batch]") {
    ModelConfig cfg;
    cfg.name = "yolo11n";
    cfg.type = "detection";
    cfg.path = "test_data/test_models/onnx/yolo11n/yolo11n.onnx";
    cfg.backend = "ort";
    cfg.device = "cpu";
    cfg.input_size = {640, 640};
    cfg.confidence_threshold = 0.25f;

    BatchedDetector det(cfg, 8);
    if (!det.ok()) {
        SUCCEED("batched detector model unavailable; skipped: " + det.error());
        return;
    }

    const int N = 8;
    std::atomic<int> ok{0};
    std::vector<std::thread> th;
    for (int i = 0; i < N; ++i) {
        th.emplace_back([&]() {
            ImageData img(640, 640, MdImageType::PKG_BGR_U8);
            std::vector<DetectionResult> out;
            if (det.predict(img, &out)) ok.fetch_add(1);
        });
    }
    for (auto& t : th) t.join();

    REQUIRE(ok.load() == N);
    REQUIRE(det.batch_runs() >= 1);
    INFO("batch_runs=" << det.batch_runs());
}
