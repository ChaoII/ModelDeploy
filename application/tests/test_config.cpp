#include <catch2/catch_test_macros.hpp>
#include "config.hpp"

TEST_CASE("Config parse simple object", "[config]") {
    auto j = parse_json(R"({"name":"test","count":3,"active":true})");
    REQUIRE(j.is_object());
    REQUIRE(j.as_object().get("name")->as_string() == "test");
    REQUIRE(j.as_object().get("count")->as_int() == 3);
    REQUIRE(j.as_object().get("active")->as_bool() == true);
}

TEST_CASE("Config parse array", "[config]") {
    auto j = parse_json(R"([1,2,3])");
    REQUIRE(j.is_array());
    REQUIRE(j.as_array().size() == 3);
    REQUIRE(j.as_array()[0].as_int() == 1);
}

TEST_CASE("Config parse nested", "[config]") {
    auto j = parse_json(R"({"a":{"b":[1,2]}})");
    REQUIRE(JsonValue::get_int(j, "a/b/0") == 1);
    REQUIRE(JsonValue::get_int(j, "a/b/1") == 2);
}

TEST_CASE("Config parse with escape", "[config]") {
    auto j = parse_json(R"({"s":"hello\nworld"})");
    REQUIRE(j.as_object().get("s")->as_string() == "hello\nworld");
}

TEST_CASE("ModelConfig default values", "[config]") {
    auto j = parse_json(R"({"id":"test1","name":"test task","input_url":"rtsp://input","output_url":"rtsp://output","models":[{"name":"det","path":"/m/yolo.onnx"}]})");
    auto cfg = task_config_from_json(j);
    REQUIRE(cfg.id == "test1");
    REQUIRE(cfg.input_url == "rtsp://input");
    REQUIRE(cfg.models.size() == 1);
    REQUIRE(cfg.models[0].name == "det");
    REQUIRE(cfg.models[0].type == "detection");          // default
    REQUIRE(cfg.models[0].confidence_threshold == 0.5f); // default
    REQUIRE(cfg.models[0].input_size[0] == 640);
    REQUIRE(cfg.models[0].input_size[1] == 640);
    REQUIRE(cfg.models[0].interval == 1);
}

TEST_CASE("TaskConfig validation", "[config]") {
    std::string err;

    TaskConfig empty;
    REQUIRE_FALSE(empty.validate(&err));
    REQUIRE(err.find("input_url") != std::string::npos);

    TaskConfig no_model;
    no_model.input_url = "rtsp://in";
    no_model.output_url = "rtsp://out";
    REQUIRE_FALSE(no_model.validate(&err));
    REQUIRE(err.find("model") != std::string::npos);

    TaskConfig ok;
    ok.input_url = "rtsp://in";
    ok.output_url = "rtsp://out";
    ok.models.push_back({"det", "detection", "/m/onnx"});
    REQUIRE(ok.validate(&err));
}

TEST_CASE("TaskConfig roundtrip JSON", "[config]") {
    TaskConfig cfg;
    cfg.id = "test-round";
    cfg.name = "roundtrip";
    cfg.input_url = "rtsp://in";
    cfg.output_url = "rtsp://out";
    cfg.models.push_back({"det1", "detection", "/m/yolo.onnx"});
    cfg.models.push_back({"face", "face_detection", "/m/scrfd.onnx"});

    auto j = task_config_to_json(cfg);
    auto cfg2 = task_config_from_json(j);

    REQUIRE(cfg2.id == cfg.id);
    REQUIRE(cfg2.input_url == cfg.input_url);
    REQUIRE(cfg2.models.size() == 2);
    REQUIRE(cfg2.models[0].name == "det1");
    REQUIRE(cfg2.models[1].name == "face");
}

TEST_CASE("Config array of models with all fields", "[config]") {
    auto j = parse_json(R"({"id":"full","input_url":"rtsp://in","output_url":"rtsp://out","models":[{"name":"p1","type":"detection","path":"/m/yolo.onnx","backend":"trt","device":"gpu","confidence_threshold":0.7,"input_size":[1280,1280],"roi":[100,200,500,500],"interval":3,"labels":["person","car"]}]})");
    auto cfg = task_config_from_json(j);
    REQUIRE(cfg.models[0].backend == "trt");
    REQUIRE(cfg.models[0].confidence_threshold == 0.7f);
    REQUIRE(cfg.models[0].input_size[0] == 1280);
    REQUIRE(cfg.models[0].roi[2] == 500);
    REQUIRE(cfg.models[0].interval == 3);
    REQUIRE(cfg.models[0].labels.size() == 2);
    REQUIRE(cfg.models[0].labels[0] == "person");
}
