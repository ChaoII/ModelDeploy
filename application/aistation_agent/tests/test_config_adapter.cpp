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
      "sensitivity": 60,
      "alarm_interval_sec": 30,
      "preview": {"enabled": false, "format": "flv"},
      "events": {
        "transport": "http",
        "http": {"url": "http://cloud/cb", "token": "tk"},
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
