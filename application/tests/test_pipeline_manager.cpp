#include <catch2/catch_test_macros.hpp>
#include "pipeline_manager.hpp"

TEST_CASE("Manager empty on construction", "[manager]") {
    PipelineManager mgr;
    REQUIRE(mgr.task_count() == 0);
    REQUIRE(mgr.list_tasks().empty());
}

TEST_CASE("Manager create task", "[manager]") {
    PipelineManager mgr;
    TaskConfig cfg;
    cfg.id = "cam1";
    cfg.input_url = "rtsp://cam1";
    cfg.output_url = "rtsp://out1";
    REQUIRE(mgr.create_task(cfg));
    REQUIRE(mgr.task_count() == 1);
}

TEST_CASE("Manager create duplicate task", "[manager]") {
    PipelineManager mgr;
    TaskConfig cfg;
    cfg.id = "cam1";
    REQUIRE(mgr.create_task(cfg));
    REQUIRE_FALSE(mgr.create_task(cfg));
}

TEST_CASE("Manager create with empty id", "[manager]") {
    PipelineManager mgr;
    TaskConfig cfg;
    REQUIRE_FALSE(mgr.create_task(cfg));
}

TEST_CASE("Manager remove task", "[manager]") {
    PipelineManager mgr;
    TaskConfig cfg;
    cfg.id = "cam1";
    REQUIRE(mgr.create_task(cfg));
    REQUIRE(mgr.remove_task("cam1"));
    REQUIRE(mgr.task_count() == 0);
}

TEST_CASE("Manager remove nonexistent", "[manager]") {
    PipelineManager mgr;
    REQUIRE_FALSE(mgr.remove_task("nonexistent"));
}

TEST_CASE("Manager start/stop task", "[manager]") {
    PipelineManager mgr;
    TaskConfig cfg;
    cfg.id = "cam1";
    cfg.input_url = "rtsp://dummy";
    cfg.output_url = "rtsp://out";
    REQUIRE(mgr.create_task(cfg));
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
    TaskConfig cfg1, cfg2;
    cfg1.id = "cam1"; cfg1.name = "Camera 1";
    cfg2.id = "cam2"; cfg2.name = "Camera 2";
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
    TaskConfig cfg;
    cfg.id = "cam1";
    cfg.name = "Front Door";
    cfg.input_url = "rtsp://front";
    cfg.output_url = "rtsp://front-out";
    REQUIRE(mgr.create_task(cfg));

    TaskConfig retrieved;
    REQUIRE(mgr.get_task_config("cam1", &retrieved));
    REQUIRE(retrieved.id == "cam1");
    REQUIRE(retrieved.name == "Front Door");
    REQUIRE(retrieved.input_url == "rtsp://front");
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
    TaskConfig cfg;
    cfg.id = "cam1";
    REQUIRE(mgr.create_task(cfg));

    ModelConfig mcfg;
    mcfg.name = "det";
    mcfg.path = "/nonexistent.onnx";

    // Model ops on a stopped task should work
    REQUIRE_FALSE(mgr.add_model("cam1", mcfg));       // no infer_group = false
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
    TaskConfig cfg1, cfg2;
    cfg1.id = "cam1"; cfg2.id = "cam2";
    REQUIRE(mgr.create_task(cfg1));
    REQUIRE(mgr.create_task(cfg2));
    REQUIRE(mgr.task_count() == 2);
    mgr.stop_all();
    REQUIRE(mgr.task_count() == 0);
}

TEST_CASE("Manager multiple create and remove", "[manager]") {
    PipelineManager mgr;
    for (int i = 0; i < 5; i++) {
        TaskConfig cfg;
        cfg.id = "cam" + std::to_string(i);
        REQUIRE(mgr.create_task(cfg));
    }
    REQUIRE(mgr.task_count() == 5);

    REQUIRE(mgr.remove_task("cam2"));
    REQUIRE(mgr.task_count() == 4);
    REQUIRE_FALSE(mgr.remove_task("cam2")); // already removed
}
