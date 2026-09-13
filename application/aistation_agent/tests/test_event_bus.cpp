#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>
#include <set>
#include "event_bus.hpp"

using modeldeploy::vision::DetectionResult;
using Catch::Approx;

static DetectionBox box(const std::string& label, int id, float score) {
    DetectionBox b; b.label_name = label; b.label_id = id; b.score = score;
    b.x = 64; b.y = 128; b.w = 96; b.h = 192;   // 像素，帧 640x640 -> 归一化 0.1,0.2,0.15,0.3
    return b;
}

TEST_CASE("EventBus assembles normalized event", "[agent][events]") {
    EventBus bus;
    std::vector<DetectionEvent> got;
    bus.set_sink([&](const DetectionEvent& e) { got.push_back(e); });
    EventMeta meta; meta.edge_code = "edge-01"; meta.camera_id = 7; meta.task_id_num = 123;
    meta.algorithm_type = "INTRUSION"; meta.alarm_interval_sec = 0;
    bus.register_task("123", meta);

    bus.on_detections("123", {box("person", 0, 0.91f)}, 640, 640, 12.3);
    REQUIRE(got.size() == 1);
    REQUIRE(got[0].edge_code == "edge-01");
    REQUIRE(got[0].camera_id == 7);
    REQUIRE(got[0].task_id == 123);
    REQUIRE(got[0].algorithm_type == "INTRUSION");
    REQUIRE(got[0].schema_version == 1);
    REQUIRE(got[0].event_id.size() == 36);
    REQUIRE(got[0].ts.find('T') != std::string::npos);
    REQUIRE(got[0].ts.back() == 'Z');
    REQUIRE(got[0].detections.size() == 1);
    REQUIRE(got[0].detections[0].label == "person");
    REQUIRE(got[0].detections[0].x == Approx(0.1f));
    REQUIRE(got[0].detections[0].y == Approx(0.2f));
    REQUIRE(got[0].detections[0].w == Approx(0.15f));
    REQUIRE(got[0].detections[0].h == Approx(0.3f));
    REQUIRE(got[0].latency_ms == Approx(12.3));
}

TEST_CASE("EventBus throttles per label", "[agent][events]") {
    EventBus bus;
    int count = 0;
    bus.set_sink([&](const DetectionEvent&) { ++count; });
    EventMeta meta; meta.alarm_interval_sec = 30;
    bus.register_task("t", meta);

    bus.on_detections("t", {box("person", 0, 0.9f)}, 640, 640, 1.0);
    bus.on_detections("t", {box("person", 0, 0.9f)}, 640, 640, 1.0);   // 同 label 被节流
    bus.on_detections("t", {box("car", 1, 0.9f)}, 640, 640, 1.0);      // 不同 label 放行
    REQUIRE(count == 2);
}

TEST_CASE("EventBus skips invalid frame dims and unregistered tasks", "[agent][events]") {
    EventBus bus;
    int count = 0;
    bus.set_sink([&](const DetectionEvent&) { ++count; });
    bus.register_task("t", EventMeta{});
    bus.on_detections("t", {box("person", 0, 0.9f)}, 0, 640, 1.0);
    bus.on_detections("ghost", {box("person", 0, 0.9f)}, 640, 640, 1.0);
    REQUIRE(count == 0);
}

TEST_CASE("EventBus uuid uniqueness", "[agent][events]") {
    std::set<std::string> ids;
    for (int i = 0; i < 100; ++i) ids.insert(EventBus::make_uuid_v4());
    REQUIRE(ids.size() == 100);
}
