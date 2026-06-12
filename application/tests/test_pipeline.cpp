#include <catch2/catch_test_macros.hpp>
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

TEST_CASE("Pipeline start without decoder fails", "[pipeline]") {
    TaskConfig cfg;
    cfg.id = "fail_test";
    cfg.input_url = "rtsp://nonexistent";
    cfg.output_url = "rtsp://dummy-out";
    Pipeline pipe(cfg);
    REQUIRE_FALSE(pipe.start());
    pipe.stop();
}

TEST_CASE("Pipeline double start", "[pipeline]") {
    TaskConfig cfg;
    cfg.id = "double";
    Pipeline pipe(cfg);
    REQUIRE_NOTHROW(pipe.start());
    REQUIRE_NOTHROW(pipe.start());
    pipe.stop();
}

TEST_CASE("Pipeline stop without start", "[pipeline]") {
    TaskConfig cfg;
    cfg.id = "safe_stop";
    Pipeline pipe(cfg);
    REQUIRE_NOTHROW(pipe.stop());
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
