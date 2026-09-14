#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>
#include "config_adapter.hpp"
#include "nlohmann/json.hpp"

using nlohmann::json;
using Catch::Approx;

static json make_task_json() {
    return json::parse(R"({
      "task_id": 123,
      "camera": {"id": 7, "name": "北门", "url": "rtsp://cam/1", "transport": "tcp"},
      "models": [
        {"name": "aistation-det", "type": "det", "backend": "ort", "device": "cpu",
         "path": "test_data/test_models/onnx/yolo11n/yolo11n_nms.onnx",
         "labels": ["person","car"], "input_size": [640,640], "confidence_threshold": 0.45}
      ],
      "roi": [[0.1,0.1],[0.9,0.1],[0.9,0.9],[0.1,0.9]],
      "schedule": {"slots": [{"day": 0, "start": 8, "end": 18}]},
      "sensitivity": 60,
      "alarm_interval_sec": 30,
      "preview": {"enabled": false, "format": "flv"},
      "events": {
        "transport": "http",
        "mqtt": {"broker": "mqtt://broker:1883", "topic": "aistation/default/edge/edge-01/camera/7/detect",
                 "qos": 1, "client_id": "aistation-agent-edge-01", "username": "u", "password": "p"},
        "http": {"url": "http://cloud/cb", "token": "tk"},
        "snapshot": {"enabled": true, "inline": true, "quality": 60, "max_width": 480},
        "buffer": {"dir": "./events_buffer", "max_mb": 512}
      },
      "decoder": {"hw_accel": "none", "device_only": false, "rtsp_transport": "tcp"},
      "encoder": {"codec": "libx264", "format": "flv", "bitrate_kbps": 2500}
    })");
}

TEST_CASE("ConfigAdapter maps task fields", "[agent][config]") {
    ConfigAdapter adapter;
    AdaptedTask t;
    std::string err;
    REQUIRE(adapter.from_json(make_task_json(), &t, &err));
    REQUIRE(t.sdk.id == "123");
    REQUIRE(t.sdk.input_url == "rtsp://cam/1");
    REQUIRE(t.sdk.decoder.rtsp_transport == "tcp");
    REQUIRE(t.camera.id == 7);
    REQUIRE(t.alarm_interval_sec == 30);
    REQUIRE(t.algorithm_type.empty());
    REQUIRE(t.events.transport == "http");
    REQUIRE(t.events.http_url == "http://cloud/cb");
    REQUIRE(t.events.http_token == "tk");
    REQUIRE(t.events.buffer_dir == "./events_buffer");
    REQUIRE(t.events.buffer_max_mb == 512);
    REQUIRE(t.sdk.enable_preview == false);
    REQUIRE(t.sdk.models.size() == 1);
    const auto& m = t.sdk.models[0];
    REQUIRE(m.name == "aistation-det");
    REQUIRE(m.type == "detection");                 // det 归一化为 detection
    REQUIRE(m.backend == "ort");
    REQUIRE(m.device == "cpu");
    REQUIRE(m.confidence_threshold == 0.45f);
    REQUIRE(m.labels.size() == 2);
    REQUIRE(m.path == "test_data/test_models/onnx/yolo11n/yolo11n_nms.onnx");
    // roi 多边形外接矩形归一化：x0=0.1,y0=0.1,x1=0.9,y1=0.9 -> x,y,w,h
    REQUIRE(m.roi_norm.size() == 4);
    REQUIRE(m.roi_norm[0] == Approx(0.1f));
    REQUIRE(m.roi_norm[1] == Approx(0.1f));
    REQUIRE(m.roi_norm[2] == Approx(0.8f));
    REQUIRE(m.roi_norm[3] == Approx(0.8f));
}

TEST_CASE("ConfigAdapter rejects missing camera url", "[agent][config]") {
    auto j = make_task_json();
    j["camera"].erase("url");
    ConfigAdapter adapter;
    AdaptedTask t;
    std::string err;
    REQUIRE_FALSE(adapter.from_json(j, &t, &err));
    REQUIRE_FALSE(err.empty());
}

TEST_CASE("ConfigAdapter rejects remote model url without fetcher", "[agent][config]") {
    auto j = make_task_json();
    j["models"][0].erase("path");
    j["models"][0]["url"] = "http://host/yolo.onnx";
    ConfigAdapter adapter;
    AdaptedTask t;
    std::string err;
    REQUIRE_FALSE(adapter.from_json(j, &t, &err));
}

TEST_CASE("ConfigAdapter clamps out-of-range roi", "[agent][config]") {
    auto j = make_task_json();
    j["roi"] = json::array({json::array({-0.5, -0.2}), json::array({1.5, 1.4})});
    ConfigAdapter adapter;
    AdaptedTask t;
    std::string err;
    REQUIRE(adapter.from_json(j, &t, &err));
    const auto& rn = t.sdk.models[0].roi_norm;
    REQUIRE(rn.size() == 4);
    REQUIRE(rn[0] == Approx(0.f));
    REQUIRE(rn[1] == Approx(0.f));
    REQUIRE(rn[2] == Approx(1.f));
    REQUIRE(rn[3] == Approx(1.f));
}

TEST_CASE("ConfigAdapter parses pedestrian_attribute pipeline", "[agent][config]") {
    ConfigAdapter adapter;
    AdaptedTask t;
    std::string err;
    json j = json::parse(R"({
      "task_id": 9,
      "camera": {"id": 1, "name": "c", "url": "test_data/test_video60.mp4"},
      "models": [{
        "name": "工作服", "type": "pedestrian_attribute", "backend": "ort", "device": "cpu",
        "det_url": "test_data/test_models/onnx/zhgd_det.onnx",
        "cls_url": "test_data/test_models/onnx/zhgd_ml.onnx",
        "attributes": ["safety_helmet","reflective_vest","safety_rope","work_uniform"],
        "labels": ["person"], "input_size": [1280,1280], "confidence_threshold": 0.5,
        "password": "secret"
      }],
      "events": {"transport": "http", "http": {"url": "http://x/cb", "token": "t"}}
    })");
    REQUIRE(adapter.from_json(j, &t, &err));
    const auto& m = t.sdk.models[0];
    REQUIRE(m.type == "pedestrian_attribute");
    REQUIRE(m.path == "test_data/test_models/onnx/zhgd_det.onnx");
    REQUIRE(m.rec_path == "test_data/test_models/onnx/zhgd_ml.onnx");
    REQUIRE(m.attributes.size() == 4);
    REQUIRE(m.attributes[3] == "work_uniform");
    REQUIRE(m.password == "secret");
}

TEST_CASE("ConfigAdapter parses ocr pipeline", "[agent][config]") {
    ConfigAdapter adapter;
    AdaptedTask t;
    std::string err;
    json j = json::parse(R"({
      "task_id": 11,
      "camera": {"id": 1, "name": "c", "url": "test_data/test_video60.mp4"},
      "models": [{
        "name": "ocr", "type": "ocr", "backend": "ort", "device": "cpu",
        "det_url": "test_data/test_models/onnx/ocr_det.onnx",
        "cls_url": "test_data/test_models/onnx/ocr_cls.onnx",
        "rec_url": "test_data/test_models/onnx/ocr_rec.onnx",
        "dict_url": "test_data/test_models/ocr/dict.txt"
      }],
      "events": {"transport": "http", "http": {"url": "http://x/cb", "token": "t"}}
    })");
    REQUIRE(adapter.from_json(j, &t, &err));
    const auto& m = t.sdk.models[0];
    REQUIRE(m.type == "ocr");
    REQUIRE(m.path == "test_data/test_models/onnx/ocr_det.onnx");
    REQUIRE(m.cls_path == "test_data/test_models/onnx/ocr_cls.onnx");
    REQUIRE(m.rec_path == "test_data/test_models/onnx/ocr_rec.onnx");
    REQUIRE(m.dict_path == "test_data/test_models/ocr/dict.txt");
}

TEST_CASE("ModelConfig ocr cls_path/dict_path roundtrip", "[agent][config]") {
    TaskConfig cfg;
    cfg.id = "ocr-round";
    ModelConfig m;
    m.name = "ocr";
    m.type = "ocr";
    m.path = "/m/det.onnx";
    m.cls_path = "/m/cls.onnx";
    m.rec_path = "/m/rec.onnx";
    m.dict_path = "/m/dict.txt";
    cfg.models.push_back(m);
    auto j = task_config_to_json(cfg);
    auto back = task_config_from_json(j);
    REQUIRE(back.models.size() == 1);
    REQUIRE(back.models[0].cls_path == "/m/cls.onnx");
    REQUIRE(back.models[0].dict_path == "/m/dict.txt");
}

TEST_CASE("ConfigAdapter parses schedule, roi polygon, snapshot and normalizes broker", "[agent][config]") {
    ConfigAdapter adapter;
    AdaptedTask t;
    std::string err;
    REQUIRE(adapter.from_json(make_task_json(), &t, &err));
    REQUIRE(t.schedule.slots.size() == 1);
    REQUIRE(t.schedule.slots[0].day == 0);
    REQUIRE(t.roi_polygon.size() == 4);
    REQUIRE(t.roi_polygon[0][0] == Approx(0.1f));
    REQUIRE(t.snapshot.enabled);
    REQUIRE(t.snapshot.inline_snapshot);
    REQUIRE(t.snapshot.quality == 60);
    REQUIRE(t.snapshot.max_width == 480);
    REQUIRE(t.events.mqtt_broker == "tcp://broker:1883");
    REQUIRE(t.events.mqtt_username == "u");
    REQUIRE(t.events.mqtt_password == "p");
}
