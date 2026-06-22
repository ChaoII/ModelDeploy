#include <catch2/catch_test_macros.hpp>
#include "pipeline_manager.hpp"

static TaskConfig make_valid_cfg(const std::string& id) {
    TaskConfig cfg;
    cfg.id = id;
    cfg.name = id;
    cfg.input_url = "rtsp://cam";
    cfg.output_url = "rtsp://out";
    ModelConfig m;
    m.name = "det";
    m.type = "detection";
    m.path = "/nonexistent.onnx";
    m.input_size = {640, 640};
    m.roi = {0, 0, 0, 0};
    m.interval = 1;
    cfg.models.push_back(m);
    return cfg;
}

TEST_CASE("Manager empty on construction", "[manager]") {
    PipelineManager mgr;
    REQUIRE(mgr.task_count() == 0);
    REQUIRE(mgr.list_tasks().empty());
}

TEST_CASE("Manager create task", "[manager]") {
    PipelineManager mgr;
    REQUIRE(mgr.create_task(make_valid_cfg("cam1")));
    REQUIRE(mgr.task_count() == 1);
}

TEST_CASE("Manager create duplicate task", "[manager]") {
    PipelineManager mgr;
    auto cfg = make_valid_cfg("cam1");
    REQUIRE(mgr.create_task(cfg));
    REQUIRE_FALSE(mgr.create_task(cfg));
}

TEST_CASE("Manager create with empty id", "[manager]") {
    PipelineManager mgr;
    TaskConfig cfg;
    REQUIRE_FALSE(mgr.create_task(cfg));
}

TEST_CASE("Manager create invalid config rejected", "[manager]") {
    PipelineManager mgr;
    TaskConfig cfg;
    cfg.id = "bad";
    // 缺少模型
    REQUIRE_FALSE(mgr.create_task(cfg));
}

TEST_CASE("Manager remove task", "[manager]") {
    PipelineManager mgr;
    REQUIRE(mgr.create_task(make_valid_cfg("cam1")));
    REQUIRE(mgr.remove_task("cam1"));
    REQUIRE(mgr.task_count() == 0);
}

TEST_CASE("Manager remove nonexistent", "[manager]") {
    PipelineManager mgr;
    REQUIRE_FALSE(mgr.remove_task("nonexistent"));
}

TEST_CASE("Manager start/stop task", "[manager]") {
    PipelineManager mgr;
    REQUIRE(mgr.create_task(make_valid_cfg("cam1")));
    // start will fail (no real stream), but should not crash
    mgr.start_task("cam1");
    mgr.stop_task("cam1");
}

TEST_CASE("Manager start nonexistent", "[manager]") {
    PipelineManager mgr;
    REQUIRE_FALSE(mgr.start_task("nope"));
}

TEST_CASE("Manager stop nonexistent", "[manager]") {
    PipelineManager mgr;
    REQUIRE_FALSE(mgr.stop_task("nope"));
}

TEST_CASE("Manager list_tasks", "[manager]") {
    PipelineManager mgr;
    auto cfg1 = make_valid_cfg("cam1");
    cfg1.name = "Camera 1";
    auto cfg2 = make_valid_cfg("cam2");
    cfg2.name = "Camera 2";
    REQUIRE(mgr.create_task(cfg1));
    REQUIRE(mgr.create_task(cfg2));

    auto tasks = mgr.list_tasks();
    REQUIRE(tasks.size() == 2);
    REQUIRE((tasks[0].id == "cam1" || tasks[1].id == "cam1"));
    REQUIRE((tasks[0].id == "cam2" || tasks[1].id == "cam2"));
    REQUIRE_FALSE(tasks[0].running);
    REQUIRE_FALSE(tasks[1].running);
}

TEST_CASE("Manager get_task_config", "[manager]") {
    PipelineManager mgr;
    auto cfg = make_valid_cfg("cam1");
    cfg.name = "Front Door";
    REQUIRE(mgr.create_task(cfg));

    TaskConfig retrieved;
    REQUIRE(mgr.get_task_config("cam1", &retrieved));
    REQUIRE(retrieved.id == "cam1");
    REQUIRE(retrieved.name == "Front Door");
    REQUIRE(retrieved.input_url == "rtsp://cam");
}

TEST_CASE("Manager get_task_config nonexistent", "[manager]") {
    PipelineManager mgr;
    TaskConfig cfg;
    REQUIRE_FALSE(mgr.get_task_config("nope", &cfg));
}

TEST_CASE("Manager get_task_config null output", "[manager]") {
    PipelineManager mgr;
    REQUIRE_FALSE(mgr.get_task_config("any", nullptr));
}

TEST_CASE("Manager model operations", "[manager]") {
    PipelineManager mgr;
    REQUIRE(mgr.create_task(make_valid_cfg("cam1")));

    ModelConfig mcfg;
    mcfg.name = "det";
    mcfg.path = "/nonexistent.onnx";
    mcfg.input_size = {640, 640};
    mcfg.roi = {0, 0, 0, 0};
    mcfg.interval = 1;

    // Model ops on a stopped task should return false (no infer_group yet)
    REQUIRE_FALSE(mgr.add_model("cam1", mcfg));
    REQUIRE_FALSE(mgr.remove_model("cam1", "det"));
    REQUIRE_FALSE(mgr.update_model("cam1", "det", mcfg));
}

TEST_CASE("Manager model operations nonexistent task", "[manager]") {
    PipelineManager mgr;
    ModelConfig mcfg;
    REQUIRE_FALSE(mgr.add_model("nope", mcfg));
    REQUIRE_FALSE(mgr.remove_model("nope", "det"));
    REQUIRE_FALSE(mgr.update_model("nope", "det", mcfg));
}

TEST_CASE("Manager stop_all", "[manager]") {
    PipelineManager mgr;
    REQUIRE(mgr.create_task(make_valid_cfg("cam1")));
    REQUIRE(mgr.create_task(make_valid_cfg("cam2")));
    REQUIRE(mgr.task_count() == 2);
    mgr.stop_all();
    REQUIRE(mgr.task_count() == 0);
}

TEST_CASE("Manager multiple create and remove", "[manager]") {
    PipelineManager mgr;
    for (int i = 0; i < 5; i++) {
        REQUIRE(mgr.create_task(make_valid_cfg("cam" + std::to_string(i))));
    }
    REQUIRE(mgr.task_count() == 5);

    REQUIRE(mgr.remove_task("cam2"));
    REQUIRE(mgr.task_count() == 4);
    REQUIRE_FALSE(mgr.remove_task("cam2")); // already removed
}

TEST_CASE("Manager model library CRUD", "[manager]") {
    PipelineManager mgr;
    ModelConfig m;
    m.name = "yolo11n";
    m.type = "detection";
    m.path = "/models/yolo.onnx";
    m.input_size = {640, 640};
    m.roi = {0, 0, 0, 0};
    m.interval = 1;

    REQUIRE(mgr.add_model_to_library(m));
    REQUIRE(mgr.list_models().size() == 1);

    ModelConfig got;
    REQUIRE(mgr.get_model("yolo11n", &got));
    REQUIRE(got.path == "/models/yolo.onnx");

    m.path = "/models/yolo_v2.onnx";
    REQUIRE(mgr.update_model_in_library("yolo11n", m));
    REQUIRE(mgr.get_model("yolo11n", &got));
    REQUIRE(got.path == "/models/yolo_v2.onnx");

    REQUIRE(mgr.remove_model_from_library("yolo11n"));
    REQUIRE(mgr.list_models().empty());
}
