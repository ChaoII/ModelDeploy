#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>
#include <ctime>
#include <set>
#include "agent_util.hpp"
#include "capability.hpp"
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
    REQUIRE(got[0].schema_version == 2);
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

TEST_CASE("EventBus drops detections outside schedule window", "[agent][events]") {
    EventBus bus;
    int count = 0;
    bus.set_sink([&](const DetectionEvent&) { ++count; });
    EventMeta meta; meta.edge_code = "e"; meta.camera_id = 1; meta.task_id_num = 1;
    std::time_t t = std::time(nullptr);
    std::tm tm{};
#if defined(_WIN32)
    localtime_s(&tm, &t);
#else
    localtime_r(&t, &tm);
#endif
    const int today = (tm.tm_wday + 6) % 7;
    meta.schedule.slots.push_back({(today + 1) % 7, 0, 24});  // 只允许"明天" -> 当前必被拒
    bus.register_task("1", meta);
    bus.on_detections("1", {box("person", 0, 0.9f)}, 640, 640, 1.0);
    REQUIRE(count == 0);

    EventMeta open_meta; open_meta.edge_code = "e"; open_meta.camera_id = 1; open_meta.task_id_num = 2;
    open_meta.schedule.slots.push_back({today, 0, 24});
    bus.register_task("2", open_meta);
    bus.on_detections("2", {box("person", 0, 0.9f)}, 640, 640, 1.0);
    REQUIRE(count == 1);
}

TEST_CASE("EventBus filters detections by roi polygon", "[agent][events]") {
    EventBus bus;
    std::vector<DetectionEvent> got;
    bus.set_sink([&](const DetectionEvent& e) { got.push_back(e); });
    EventMeta meta; meta.edge_code = "e"; meta.camera_id = 1; meta.task_id_num = 1;
    meta.roi_polygon = {{{0.0f,0.0f}},{{0.5f,0.0f}},{{0.5f,0.5f}},{{0.0f,0.5f}}};  // 左上象限
    bus.register_task("1", meta);
    DetectionBox inside = box("person", 0, 0.9f);   // 中心 (0.2,0.4) 落在多边形内
    inside.x = 32; inside.y = 160; inside.w = 192; inside.h = 192;
    DetectionBox outside = box("person", 0, 0.9f);  // 中心 (0.8,0.8) 在多边形外
    outside.x = 448; outside.y = 448; outside.w = 128; outside.h = 128;
    bus.on_detections("1", {inside, outside}, 640, 640, 1.0);
    REQUIRE(got.size() == 1);
    REQUIRE(got[0].detections.size() == 1);
}

TEST_CASE("EventBus carries object attributes", "[agent][events]") {
    EventBus bus;
    std::vector<DetectionEvent> got;
    bus.set_sink([&](const DetectionEvent& e) { got.push_back(e); });
    EventMeta meta; meta.edge_code = "e"; meta.camera_id = 1; meta.task_id_num = 1;
    bus.register_task("1", meta);
    DetectionBox b = box("person", 0, 0.9f);
    b.attributes = {{"work_uniform", 0.2f}};
    bus.on_detections("1", {b}, 640, 640, 1.0);
    REQUIRE(got.size() == 1);
    REQUIRE(got[0].detections.size() == 1);
    REQUIRE(got[0].detections[0].attributes.at("work_uniform") == Approx(0.2f));
    // 事件 v2：objects 序列化（含属性），detections 兼容保留
    auto j = got[0].to_json();
    REQUIRE(j["schema_version"] == 2);
    REQUIRE(j.contains("objects"));
    REQUIRE(j["objects"][0]["attributes"]["work_uniform"] == Approx(0.2f));
    REQUIRE(j.contains("detections"));
}

TEST_CASE("EventBus round-trip preserves attributes", "[agent][events]") {
    DetectionEvent e;
    e.event_id = "rt"; e.ts = "2026-01-01T00:00:00.000Z";
    EventDetection d; d.label = "person"; d.attributes["work_uniform"] = 0.2f;
    e.detections.push_back(d);
    DetectionEvent back = DetectionEvent::from_json(e.to_json());
    REQUIRE(back.detections.size() == 1);
    REQUIRE(back.detections[0].attributes.at("work_uniform") == Catch::Approx(0.2f));
}

TEST_CASE("EventBus carries OCR text", "[agent][events]") {
    EventBus bus;
    std::vector<DetectionEvent> got;
    bus.set_sink([&](const DetectionEvent& e) { got.push_back(e); });
    EventMeta meta; meta.edge_code = "e"; meta.camera_id = 1; meta.task_id_num = 1;
    bus.register_task("1", meta);
    DetectionBox b = box("text", 0, 0.9f);
    b.text = "沪A12345"; b.text_score = 0.88f;
    bus.on_detections("1", {b}, 640, 640, 1.0);
    REQUIRE(got.size() == 1);
    REQUIRE(got[0].detections.size() == 1);
    REQUIRE(got[0].detections[0].text == "沪A12345");
    REQUIRE(got[0].detections[0].text_score == Approx(0.88f));
    // 事件 v2：objects 序列化携带 text/text_score
    auto j = got[0].to_json();
    REQUIRE(j["objects"][0]["text"] == "沪A12345");
    REQUIRE(j["objects"][0]["text_score"] == Approx(0.88f));
    // 无文字的 box 不写 text 字段
    DetectionEvent e;
    EventDetection empty;
    empty.label = "person";
    e.detections.push_back(empty);
    REQUIRE_FALSE(e.to_json()["objects"][0].contains("text"));
}

TEST_CASE("EventBus round-trip preserves text", "[agent][events]") {
    DetectionEvent e;
    e.event_id = "rt2"; e.ts = "2026-01-01T00:00:00.000Z";
    EventDetection d; d.label = "text"; d.text = "hello"; d.text_score = 0.5f;
    e.detections.push_back(d);
    DetectionEvent back = DetectionEvent::from_json(e.to_json());
    REQUIRE(back.detections.size() == 1);
    REQUIRE(back.detections[0].text == "hello");
    REQUIRE(back.detections[0].text_score == Approx(0.5f));
}

TEST_CASE("EventBus carries LPR plate text", "[agent][events]") {
    EventBus bus;
    std::vector<DetectionEvent> got;
    bus.set_sink([&](const DetectionEvent& e) { got.push_back(e); });
    EventMeta meta; meta.edge_code = "e"; meta.camera_id = 1; meta.task_id_num = 1;
    bus.register_task("1", meta);
    // LPR：label_name="plate"、label_id=-1、text=车牌号（InferenceEngine::infer_lpr 产出形态）
    DetectionBox b = box("plate", -1, 0.95f);
    b.text = "沪A12345"; b.text_score = 0.95f;
    bus.on_detections("1", {b}, 640, 640, 1.0);
    REQUIRE(got.size() == 1);
    REQUIRE(got[0].detections.size() == 1);
    REQUIRE(got[0].detections[0].label == "plate");
    REQUIRE(got[0].detections[0].text == "沪A12345");
    REQUIRE(got[0].detections[0].text_score == Approx(0.95f));
    // 事件 v2：objects 序列化携带车牌文字
    auto j = got[0].to_json();
    REQUIRE(j["objects"][0]["text"] == "沪A12345");
}

TEST_CASE("EventBus carries track_id", "[agent][events]") {
    EventBus bus;
    std::vector<DetectionEvent> got;
    bus.set_sink([&](const DetectionEvent& e) { got.push_back(e); });
    EventMeta meta; meta.edge_code = "e"; meta.camera_id = 1; meta.task_id_num = 1;
    bus.register_task("1", meta);
    DetectionBox b = box("person", 0, 0.9f);
    b.track_id = 42;
    bus.on_detections("1", {b}, 640, 640, 1.0);
    REQUIRE(got.size() == 1);
    REQUIRE(got[0].detections.size() == 1);
    REQUIRE(got[0].detections[0].track_id == 42);
    // 事件 v2：objects 序列化携带 track_id
    auto j = got[0].to_json();
    REQUIRE(j["objects"][0]["track_id"] == 42);
    // 未跟踪（-1）不写 track_id 字段
    DetectionEvent e;
    EventDetection untracked; untracked.label = "person";
    e.detections.push_back(untracked);
    REQUIRE_FALSE(e.to_json()["objects"][0].contains("track_id"));
}

TEST_CASE("EventBus round-trip preserves track_id", "[agent][events]") {
    DetectionEvent e;
    e.event_id = "rt3"; e.ts = "2026-01-01T00:00:00.000Z";
    EventDetection d; d.label = "person"; d.track_id = 7;
    e.detections.push_back(d);
    DetectionEvent back = DetectionEvent::from_json(e.to_json());
    REQUIRE(back.detections.size() == 1);
    REQUIRE(back.detections[0].track_id == 7);
    // 未跟踪事件往返后保持 -1
    DetectionEvent e2;
    EventDetection d2; d2.label = "person";
    e2.detections.push_back(d2);
    DetectionEvent back2 = DetectionEvent::from_json(e2.to_json());
    REQUIRE(back2.detections.size() == 1);
    REQUIRE(back2.detections[0].track_id == -1);
}

TEST_CASE("detect_capabilities advertises tracking family", "[agent][capability]") {
    auto c = detect_capabilities(8);
    bool has_tracking = false;
    for (const auto& f : c["model_families"]) {
        if (f == "tracking") has_tracking = true;
    }
    REQUIRE(has_tracking);
}

TEST_CASE("DetectionEvent serializes inline snapshot", "[agent][events]") {
    DetectionEvent e;
    e.event_id = "x"; e.snapshot_data = "QUJD"; e.snapshot_width = 320; e.snapshot_height = 180;
    auto j = e.to_json();
    REQUIRE(j["snapshot"]["data"] == "QUJD");
    REQUIRE(j["snapshot"]["width"] == 320);
    REQUIRE(j["snapshot"]["height"] == 180);
}
