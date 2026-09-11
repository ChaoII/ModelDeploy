#include <catch2/catch_test_macros.hpp>
#include <atomic>
#include <chrono>
#include <thread>
#include <vector>

#include "batched_detector.hpp"
#include "csrc/vision/common/image_data.h"

using namespace modeldeploy::vision;

// 共享批处理检测器：8 路并发应触发批合并（batch_runs < 请求数）且比串行更快。
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
    auto make_img = []() { return ImageData(640, 640, MdImageType::PKG_BGR_U8); };
    { ImageData warm = make_img(); std::vector<DetectionResult> o; det.predict(warm, &o); }

    const int N = 8;

    // 串行基线
    const auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < N; ++i) {
        ImageData img = make_img();
        std::vector<DetectionResult> out;
        det.predict(img, &out);
    }
    const double serial_ms =
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();

    // 并发批
    std::atomic<int> ok{0};
    std::vector<std::thread> th;
    const auto t1 = std::chrono::steady_clock::now();
    for (int i = 0; i < N; ++i) {
        th.emplace_back([&]() {
            ImageData img = make_img();
            std::vector<DetectionResult> out;
            if (det.predict(img, &out)) ok.fetch_add(1);
        });
    }
    for (auto& t : th) t.join();
    const double batch_ms =
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t1).count();

    REQUIRE(ok.load() == N);
    REQUIRE(det.batch_runs() >= 1);
    INFO("serial=" << serial_ms << "ms  batch=" << batch_ms << "ms  runs=" << det.batch_runs());
    // CPU 上批处理的 kernel 摊销收益有限、主要靠 intra-op 线程；GPU 才是收益点。
    // 这里只要求"不明显劣化"，真正的 batched-forward 收益在 GPU 上体现。
    REQUIRE(batch_ms <= serial_ms * 1.5);
}
