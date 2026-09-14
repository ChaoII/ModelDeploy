#include <catch2/catch_test_macros.hpp>
#include <atomic>
#include <chrono>
#include <fstream>
#include <thread>
#include "pipeline.hpp"

static const char* kVideo = "test_data/test_video60.mp4";
static const char* kModel = "test_data/test_models/onnx/yolo11n/yolo11n_nms.onnx";

TEST_CASE("Pipeline detection sink fires on real detections", "[agent][pipeline][integration]") {
    if (!std::ifstream(kVideo).good() || !std::ifstream(kModel).good())
        SKIP("test video/model absent");

    TaskConfig cfg;
    cfg.id = "sink_test";
    cfg.input_url = kVideo;
    cfg.enable_preview = false;
    ModelConfig m;
    m.name = "det"; m.type = "detection"; m.backend = "ort"; m.device = "cpu";
    m.path = kModel; m.labels = {"person"}; m.input_size = {640, 640};
    cfg.models.push_back(m);

    Pipeline pipe(cfg);
    std::atomic<int> calls{0};
    std::atomic<int> boxes{0};
    pipe.set_detection_sink([&](const std::vector<DetectionBox>& b, int w, int h, double) {
        ++calls; boxes += static_cast<int>(b.size());
        REQUIRE(w > 0); REQUIRE(h > 0);
    });
    REQUIRE(pipe.start());
    for (int i = 0; i < 100 && calls.load() == 0; ++i)
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
    pipe.stop();
    REQUIRE(calls.load() > 0);
    REQUIRE(boxes.load() > 0);
}

// 属性 pipeline 送达 sink 的集成用例：驱动真实 Pipeline，断言检测 sink 收到
// 携带非空 attributes 的 DetectionBox（覆盖 detect_loop 中 non_det → sink 的合并分支）。
TEST_CASE("Pipeline pedestrian_attribute delivers attributes to sink",
          "[agent][pipeline][integration]") {
    if (!std::ifstream(kVideo).good() ||
        !std::ifstream("test_data/test_models/onnx/zhgd_det.onnx").good() ||
        !std::ifstream("test_data/test_models/onnx/zhgd_ml.onnx").good())
        SKIP("test data absent");

    TaskConfig cfg;
    cfg.id = "attr_sink_test";
    cfg.input_url = kVideo;
    cfg.enable_preview = false;
    ModelConfig m;
    m.name = "pa";
    m.type = "pedestrian_attribute";
    m.backend = "ort";
    m.device = "cpu";
    m.path = "test_data/test_models/onnx/zhgd_det.onnx";
    m.rec_path = "test_data/test_models/onnx/zhgd_ml.onnx";
    m.input_size = {1280, 1280};
    m.confidence_threshold = 0.5f;
    m.attributes = {"safety_helmet", "reflective_vest", "safety_rope", "work_uniform"};
    cfg.models.push_back(m);

    Pipeline pipe(cfg);
    std::atomic<int> attr_boxes{0};
    pipe.set_detection_sink([&](const std::vector<DetectionBox>& boxes, int w, int h, double) {
        REQUIRE(w > 0);
        REQUIRE(h > 0);
        for (const auto& b : boxes)
            if (!b.attributes.empty()) ++attr_boxes;
    });
    REQUIRE(pipe.start());
    // 有界等待（模型加载 + 推理，上限 20s）
    for (int i = 0; i < 400 && attr_boxes.load() == 0; ++i)
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
    pipe.stop();
    REQUIRE(attr_boxes.load() > 0);
}

TEST_CASE("Pipeline without sink unaffected", "[agent][pipeline]") {
    TaskConfig cfg;
    cfg.id = "no_sink";
    cfg.input_url = "E:/nonexistent.mp4";
    cfg.output_url = "x.flv";
    Pipeline pipe(cfg);
    REQUIRE(pipe.start());
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    pipe.stop();
    REQUIRE_FALSE(pipe.is_running());
}
