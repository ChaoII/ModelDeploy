#include <catch2/catch_test_macros.hpp>
#include <thread>
#include <chrono>
#include "pipeline.hpp"

TEST_CASE("Pipeline construct/destroy", "[pipeline]") {
    TaskConfig cfg;
    cfg.id = "test";
    cfg.input_url = "rtsp://dummy";
    cfg.output_url = "rtsp://dummy-out";
    Pipeline pipe(cfg);
    REQUIRE_FALSE(pipe.is_running());
}

TEST_CASE("Pipeline task_id", "[pipeline]") {
    TaskConfig cfg;
    cfg.id = "cam01";
    Pipeline pipe(cfg);
    REQUIRE(pipe.task_id() == "cam01");
}

TEST_CASE("Pipeline start is async", "[pipeline]") {
    TaskConfig cfg;
    cfg.id = "async_test";
    cfg.input_url = "rtsp://nonexistent";
    cfg.output_url = "rtsp://dummy-out";
    Pipeline pipe(cfg);
    // start() 返回 true，初始化在后台进行
    bool ok = pipe.start();
    REQUIRE(ok);
    REQUIRE(pipe.is_running());
    // 初始化最终会失败（无 RTSP 流），线程会退出
    std::this_thread::sleep_for(std::chrono::milliseconds(200));
    pipe.stop();
}

TEST_CASE("Pipeline double start", "[pipeline]") {
    TaskConfig cfg;
    cfg.id = "double";
    Pipeline pipe(cfg);
    REQUIRE(pipe.start());
    REQUIRE(pipe.start());  // second start is no-op
    pipe.stop();
}

TEST_CASE("Pipeline stop without start", "[pipeline]") {
    TaskConfig cfg;
    cfg.id = "safe_stop";
    Pipeline pipe(cfg);
    REQUIRE_NOTHROW(pipe.stop());
}

TEST_CASE("Pipeline start then stop quickly", "[pipeline]") {
    TaskConfig cfg;
    cfg.id = "fast_stop";
    cfg.input_url = "rtsp://dummy";
    cfg.output_url = "rtsp://out";
    Pipeline pipe(cfg);
    REQUIRE(pipe.start());
    pipe.stop();
    REQUIRE_FALSE(pipe.is_running());
}

TEST_CASE("Pipeline model operations without start", "[pipeline]") {
    TaskConfig cfg;
    cfg.id = "model_ops";
    Pipeline pipe(cfg);

    ModelConfig mcfg;
    mcfg.name = "det";
    mcfg.path = "/nonexistent.onnx";
    REQUIRE_FALSE(pipe.add_model(mcfg));
    REQUIRE_FALSE(pipe.remove_model("det"));
    REQUIRE_FALSE(pipe.update_model("det", mcfg));
}

TEST_CASE("Pipeline config", "[pipeline]") {
    TaskConfig cfg;
    cfg.id = "config_test";
    cfg.input_url = "rtsp://cam";
    cfg.output_url = "rtmp://push";
    Pipeline pipe(cfg);
    const auto& c = pipe.config();
    REQUIRE(c.id == "config_test");
    REQUIRE(c.input_url == "rtsp://cam");
    REQUIRE(c.output_url == "rtmp://push");
}

TEST_CASE("Pipeline real model and stream", "[pipeline][integration]") {
    TaskConfig cfg;
    cfg.id = "integration_test";
    cfg.input_url = "rtsp://127.0.0.1:554/live/obs_test3";
    cfg.output_url = "E:/CLionProjects/ModelDeploy/build/bin/integration_out.mp4";
    cfg.draw.show_label = true;
    cfg.draw.show_score = true;

    ModelConfig mcfg;
    mcfg.name = "yolo11n";
    mcfg.type = "detection";
    mcfg.path = "E:/CLionProjects/ModelDeploy/test_data/test_models/yolo11n_nms.onnx";
    mcfg.backend = "ort";
    mcfg.device = "gpu";
    mcfg.confidence_threshold = 0.5f;
    mcfg.input_size = {640, 640};
    mcfg.interval = 1;
    cfg.models.push_back(mcfg);

    Pipeline pipe(cfg);
    // start() is async, returns immediately
    REQUIRE_NOTHROW(pipe.start());
    // Wait for initialization to complete or fail
    std::this_thread::sleep_for(std::chrono::seconds(3));
    pipe.stop();
}
