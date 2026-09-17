#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>
#include <cmath>
#include <cstdio>
#include <ctime>
#include <set>
#include <string>
#include <vector>
#include "agent_util.hpp"
#include "capability.hpp"
#include "config_adapter.hpp"
#include "event_bus.hpp"
#include "nlohmann/json.hpp"

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
    // 与云端 gating 一致：schedule_active_now 按 UTC 判定，故期望值也用 gmtime 推导
    std::time_t t = std::time(nullptr);
    std::tm tm{};
#if defined(_WIN32)
    gmtime_s(&tm, &t);
#else
    gmtime_r(&t, &tm);
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

TEST_CASE("EventBus emits heartbeat with empty detections", "[agent][events]") {
    EventBus bus;
    std::vector<DetectionEvent> got;
    bus.set_sink([&](const DetectionEvent& e) { got.push_back(e); });
    EventMeta meta; meta.edge_code = "edge-01"; meta.camera_id = 7; meta.task_id_num = 123;
    meta.algorithm_type = "ABSENT"; meta.alarm_interval_sec = 0;
    bus.register_task("123", meta);

    bus.on_heartbeat("123");
    REQUIRE(got.size() == 1);
    REQUIRE(got[0].heartbeat == true);
    REQUIRE(got[0].detections.empty());
    REQUIRE(got[0].edge_code == "edge-01");
    REQUIRE(got[0].camera_id == 7);
    REQUIRE(got[0].task_id == 123);
    REQUIRE(got[0].algorithm_type == "ABSENT");
    REQUIRE(!got[0].event_id.empty());
    REQUIRE(got[0].schema_version == 2);

    // 序列化后 heartbeat 标记必须出现，空数组必须存在
    const auto j = got[0].to_json();
    REQUIRE(j.value("heartbeat", false) == true);
    REQUIRE(j["detections"].is_array());
    REQUIRE(j["detections"].empty());

    // 未注册任务不发布
    bus.on_heartbeat("nope");
    REQUIRE(got.size() == 1);
}

TEST_CASE("DetectionEvent without heartbeat omits the flag", "[agent][events]") {
    EventBus bus;
    std::vector<DetectionEvent> got;
    bus.set_sink([&](const DetectionEvent& e) { got.push_back(e); });
    EventMeta meta; meta.edge_code = "edge-01"; meta.camera_id = 7; meta.task_id_num = 123;
    meta.algorithm_type = "INTRUSION"; meta.alarm_interval_sec = 0;
    bus.register_task("123", meta);
    bus.on_detections("123", {box("person", 0, 0.91f)}, 640, 640, 1.0);
    REQUIRE(got.size() == 1);
    REQUIRE(got[0].heartbeat == false);
    REQUIRE(got[0].to_json().contains("heartbeat") == false);
}

// ── face_rec 嵌入紧凑编码（f16b64）与 Top-N 上限 ──

// 黄金向量跨语言对拍：Python struct.pack('<5e') + base64 的结果
TEST_CASE("embedding f16b64 matches cross-language golden", "[agent][events]") {
    const std::vector<float> v{0.5f, -0.25f, 1.0f, 0.0f, 0.25f};
    REQUIRE(encode_embedding_f16_b64(v) == "ADgAtAA8AAAANA==");
    const auto back = decode_embedding_f16_b64("ADgAtAA8AAAANA==");
    REQUIRE(back.size() == 5);
    REQUIRE(back[0] == Approx(0.5f));
    REQUIRE(back[1] == Approx(-0.25f));
    REQUIRE(back[2] == Approx(1.0f));
    REQUIRE(back[3] == Approx(0.0f));
    REQUIRE(back[4] == Approx(0.25f));
}

// 512-d 往返：float16 精度损失后余弦仍接近 1
TEST_CASE("embedding f16b64 round-trips within float16 tolerance", "[agent][events]") {
    std::vector<float> v;
    v.reserve(512);
    for (int i = 0; i < 512; ++i) v.push_back(std::sin(static_cast<float>(i) * 0.37f));
    double n = 0.0;
    for (float x : v) n += static_cast<double>(x) * x;
    n = std::sqrt(n);
    for (auto& x : v) x = static_cast<float>(x / n);

    const auto back = decode_embedding_f16_b64(encode_embedding_f16_b64(v));
    REQUIRE(back.size() == 512);
    double dot = 0.0, na = 0.0, nb = 0.0;
    for (size_t i = 0; i < v.size(); ++i) {
        dot += static_cast<double>(v[i]) * back[i];
        na += static_cast<double>(v[i]) * v[i];
        nb += static_cast<double>(back[i]) * back[i];
    }
    REQUIRE(dot / std::sqrt(na * nb) > 0.9999);
}

TEST_CASE("embedding f16b64 rejects malformed input", "[agent][events]") {
    REQUIRE(decode_embedding_f16_b64("").empty());
    REQUIRE(decode_embedding_f16_b64("not-base64!!").empty());
    REQUIRE(decode_embedding_f16_b64("ADgA").empty());   // base64 合法但字节数为奇数
    REQUIRE(encode_embedding_f16_b64({}).empty());
}

TEST_CASE("DetectionEvent serializes embedding as f16b64 string", "[agent][events]") {
    DetectionEvent e;
    e.event_id = "enc"; e.ts = "2026-01-01T00:00:00.000Z";
    EventDetection d; d.label = "face"; d.embedding = {0.6f, 0.8f};
    e.detections.push_back(d);

    auto j = e.to_json();
    REQUIRE(j["objects"][0].contains("embedding"));
    REQUIRE(j["objects"][0]["embedding_encoding"] == "f16b64");
    REQUIRE(j["objects"][0]["embedding"].is_string());

    auto back = DetectionEvent::from_json(j);
    REQUIRE(back.detections.size() == 1);
    REQUIRE(back.detections[0].embedding.size() == 2);
    REQUIRE(back.detections[0].embedding[0] == Approx(0.6f).margin(1e-3));
    REQUIRE(back.detections[0].embedding[1] == Approx(0.8f).margin(1e-3));
}

// 向后兼容：旧格式原始 float 数组仍可解析
TEST_CASE("DetectionEvent from_json accepts legacy raw float embedding", "[agent][events]") {
    nlohmann::json j;
    j["detections"] = nlohmann::json::array();
    j["detections"].push_back({{"label", "face"}, {"bbox", nlohmann::json::object()}});
    j["objects"] = nlohmann::json::array();
    j["objects"].push_back({{"label", "face"}, {"embedding", {0.5, 0.25}}});
    auto e = DetectionEvent::from_json(j);
    REQUIRE(e.detections.size() == 1);
    REQUIRE(e.detections[0].embedding.size() == 2);
    REQUIRE(e.detections[0].embedding[1] == Approx(0.25f));
}

// 上限：12 张带嵌入的人脸 → 仅置信度最高的 8 个携带嵌入
TEST_CASE("EventBus caps embeddings to top-N by confidence", "[agent][events]") {
    EventBus bus;
    std::vector<DetectionEvent> got;
    bus.set_sink([&](const DetectionEvent& e) { got.push_back(e); });
    EventMeta meta; meta.task_id_num = 1; meta.alarm_interval_sec = 0; meta.max_embeddings = 8;
    bus.register_task("t", meta);

    std::vector<DetectionBox> boxes;
    for (int i = 0; i < 12; ++i) {
        DetectionBox b = box("face", 0, 0.1f * static_cast<float>(i + 1));
        b.embedding = {static_cast<float>(i), 1.0f};
        boxes.push_back(b);
    }
    bus.on_detections("t", boxes, 640, 640, 1.0);
    REQUIRE(got.size() == 1);

    const auto j = got[0].to_json();
    REQUIRE(j["objects"].size() == 12);
    int with_embedding = 0;
    for (const auto& o : j["objects"]) {
        if (o.contains("embedding")) ++with_embedding;
    }
    REQUIRE(with_embedding == 8);
    // 保留置信度最高的 8 个（i=4..11），低分的 4 个不带嵌入
    REQUIRE_FALSE(j["objects"][0].contains("embedding"));
    REQUIRE_FALSE(j["objects"][3].contains("embedding"));
    REQUIRE(j["objects"][4].contains("embedding"));
    REQUIRE(j["objects"][11].contains("embedding"));
    // 解码保留向量：objects[4] 的首元素应为 4.0
    const auto vec = decode_embedding_f16_b64(j["objects"][4]["embedding"].get<std::string>());
    REQUIRE(vec.size() == 2);
    REQUIRE(vec[0] == Approx(4.0f));
}

// 默认上限为 8（未显式配置时同样生效）
TEST_CASE("EventBus applies default embedding cap of 8", "[agent][events]") {
    EventBus bus;
    std::vector<DetectionEvent> got;
    bus.set_sink([&](const DetectionEvent& e) { got.push_back(e); });
    EventMeta meta; meta.task_id_num = 1; meta.alarm_interval_sec = 0;
    bus.register_task("t", meta);

    std::vector<DetectionBox> boxes;
    for (int i = 0; i < 10; ++i) {
        DetectionBox b = box("face", 0, 0.5f);
        b.embedding = {1.0f};
        boxes.push_back(b);
    }
    bus.on_detections("t", boxes, 640, 640, 1.0);
    REQUIRE(got.size() == 1);
    const auto j = got[0].to_json();
    int with_embedding = 0;
    for (const auto& o : j["objects"]) {
        if (o.contains("embedding")) ++with_embedding;
    }
    REQUIRE(with_embedding == 8);
}

// 最坏体积：24 张 512-d 人脸 + Top-8 上限，事件体积必须远低于原始 float32
TEST_CASE("embedding payload worst-case is bounded", "[agent][events]") {
    EventBus bus;
    std::vector<DetectionEvent> got;
    bus.set_sink([&](const DetectionEvent& e) { got.push_back(e); });
    EventMeta meta; meta.task_id_num = 1; meta.alarm_interval_sec = 0; meta.max_embeddings = 8;
    bus.register_task("t", meta);

    std::vector<DetectionBox> boxes;
    for (int i = 0; i < 24; ++i) {
        DetectionBox b = box("face", 0, 0.5f + 0.01f * static_cast<float>(i));
        b.embedding.reserve(512);
        for (int k = 0; k < 512; ++k) {
            b.embedding.push_back(std::sin(static_cast<float>(i * 512 + k) * 0.017f));
        }
        boxes.push_back(b);
    }
    bus.on_detections("t", boxes, 640, 640, 1.0);
    REQUIRE(got.size() == 1);
    const auto j = got[0].to_json();

    int with_embedding = 0;
    for (const auto& o : j["objects"]) {
        if (o.contains("embedding")) ++with_embedding;
    }
    REQUIRE(with_embedding == 8);

    const size_t encoded_bytes = j.dump().size();
    // 对照：单张 512-d 脸 f16b64 编码长度 = base64(1024 bytes) = 1368
    REQUIRE(encode_embedding_f16_b64(boxes[0].embedding).size() == 1368);

    // 对照：原始 float32 数组体积（每脸约 10.7KB / 24 脸 ~250KB）
    size_t raw_bytes = 0;
    for (const auto& b : boxes) {
        raw_bytes += nlohmann::json(b.embedding).dump().size();
    }
    // 最坏事件体积：f16b64 + Top-8 后远小于原始（24 脸被压到约 20KB）
    std::printf("[measure] per_face_raw=%zu per_face_f16b64=%zu encoded_event=%zu raw_embeddings=%zu\n",
                nlohmann::json(boxes[0].embedding).dump().size(),
                encode_embedding_f16_b64(boxes[0].embedding).size(), encoded_bytes, raw_bytes);
    REQUIRE(encoded_bytes < 24 * 1024);
    REQUIRE(raw_bytes > 100 * 1024);
    REQUIRE(raw_bytes > 4 * encoded_bytes);
}

// 端到端：任务配置顶层 `max_embeddings: 3` → 单事件最多 3 个对象携带嵌入
TEST_CASE("EventBus honors config max_embeddings", "[agent][events]") {
    ConfigAdapter adapter(nullptr);
    std::string err;
    auto cfg = nlohmann::json::parse(R"({
      "task_id": 123,
      "camera": {"id": 7, "name": "北门", "url": "rtsp://cam/1"},
      "models": [{"name": "det", "type": "det",
                  "path": "test_data/test_models/onnx/yolo11n/yolo11n_nms.onnx"}],
      "max_embeddings": 3
    })");
    AdaptedTask t;
    REQUIRE(adapter.from_json(cfg, &t, &err));
    REQUIRE(t.max_embeddings == 3);

    EventBus bus;
    std::vector<DetectionEvent> got;
    bus.set_sink([&](const DetectionEvent& e) { got.push_back(e); });
    EventMeta meta;
    meta.task_id_num = 1;
    meta.alarm_interval_sec = 0;
    meta.max_embeddings = t.max_embeddings;
    bus.register_task("t", meta);

    std::vector<DetectionBox> boxes;
    for (int i = 0; i < 6; ++i) {
        DetectionBox b = box("face", 0, 0.5f + 0.05f * static_cast<float>(i));
        b.embedding = {1.0f};
        boxes.push_back(b);
    }
    bus.on_detections("t", boxes, 640, 640, 1.0);
    REQUIRE(got.size() == 1);
    const auto j = got[0].to_json();
    REQUIRE(j["objects"].size() == 6);
    int with_embedding = 0;
    for (const auto& o : j["objects"]) {
        if (o.contains("embedding")) ++with_embedding;
    }
    REQUIRE(with_embedding == 3);
    // 保留置信度最高的 3 个（i=3..5），低分的 3 个不带嵌入
    REQUIRE_FALSE(j["objects"][0].contains("embedding"));
    REQUIRE(j["objects"][3].contains("embedding"));
    REQUIRE(j["objects"][5].contains("embedding"));
}

// 端到端：配置 `max_embeddings: 0` → 禁用嵌入（全部对象不带 embedding）
TEST_CASE("EventBus disables embeddings when config max_embeddings is zero", "[agent][events]") {
    ConfigAdapter adapter(nullptr);
    std::string err;
    auto cfg = nlohmann::json::parse(R"({
      "task_id": 123,
      "camera": {"id": 7, "name": "北门", "url": "rtsp://cam/1"},
      "models": [{"name": "det", "type": "det",
                  "path": "test_data/test_models/onnx/yolo11n/yolo11n_nms.onnx"}],
      "max_embeddings": 0
    })");
    AdaptedTask t;
    REQUIRE(adapter.from_json(cfg, &t, &err));
    REQUIRE(t.max_embeddings == 0);

    EventBus bus;
    std::vector<DetectionEvent> got;
    bus.set_sink([&](const DetectionEvent& e) { got.push_back(e); });
    EventMeta meta;
    meta.task_id_num = 1;
    meta.alarm_interval_sec = 0;
    meta.max_embeddings = t.max_embeddings;
    bus.register_task("t", meta);

    std::vector<DetectionBox> boxes;
    for (int i = 0; i < 4; ++i) {
        DetectionBox b = box("face", 0, 0.5f + 0.05f * static_cast<float>(i));
        b.embedding = {1.0f};
        boxes.push_back(b);
    }
    bus.on_detections("t", boxes, 640, 640, 1.0);
    REQUIRE(got.size() == 1);
    const auto j = got[0].to_json();
    for (const auto& o : j["objects"]) {
        REQUIRE_FALSE(o.contains("embedding"));
    }
}
