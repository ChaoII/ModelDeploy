#include <catch2/catch_test_macros.hpp>
#include <thread>
#include <chrono>
#include <filesystem>
#include <fstream>
#include "pipeline.hpp"

static const char* kTestMp4 = "E:/CLionProjects/ModelDeploy/test_data/bench_videos/cam00.mp4";

TEST_CASE("Pipeline construct/destroy", "[pipeline]") {
    TaskConfig cfg;
    cfg.id = "test";
    cfg.input_url = kTestMp4;
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

TEST_CASE("Pipeline start fails fast on missing file", "[pipeline]") {
    TaskConfig cfg;
    cfg.id = "missing_file";
    cfg.input_url = "E:/videos/definitely_not_here.mp4";
    cfg.output_url = "rtsp://out";
    Pipeline pipe(cfg);
    bool ok = pipe.start();
    REQUIRE(ok);
    std::this_thread::sleep_for(std::chrono::milliseconds(200));
    pipe.stop();
    REQUIRE_FALSE(pipe.is_running());
}

// 集成：本地 mp4 → .flv 输出，空模型列表（run_models 返回 false 被容忍，仍编码原帧）
TEST_CASE("Pipeline decode+encode integration", "[pipeline][integration]") {
    if (!std::ifstream(kTestMp4).good()) {
        SKIP("test mp4 absent; skipping integration");
    }
    const std::string out = std::filesystem::temp_directory_path().string()
                            + "/md_pipeline_" +
                            std::to_string(std::chrono::steady_clock::now().time_since_epoch().count())
                            + ".flv";

    TaskConfig cfg;
    cfg.id = "integration";
    cfg.input_url = kTestMp4;
    cfg.output_url = out;
    cfg.enable_preview = true;
    cfg.encoder.codec = "libx264";
    // 空模型列表：run_models 返回 false，pipeline 仍对原帧编码

    Pipeline pipe(cfg);
    REQUIRE(pipe.start());
    REQUIRE(pipe.is_running());

    int64_t c0 = pipe.stats().frame_count;
    std::this_thread::sleep_for(std::chrono::seconds(3));
    int64_t c1 = pipe.stats().frame_count;

    REQUIRE(pipe.is_running());
    REQUIRE(c1 > c0);

    pipe.stop();
    REQUIRE_FALSE(pipe.is_running());
    REQUIRE(std::filesystem::exists(out));
    REQUIRE(std::filesystem::file_size(out) > 0);
}
