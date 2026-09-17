// pose 家族接线用例：
// 1) 类型名归一化（云端场景目录声明 type="pose" + 别名）
// 2) 能力上报含 pose
// 3) 真机推理：yolo11n-pose 输出人体框 + 每人 17 个关键点（像素坐标、帧内）
// 4) 事件契约：关键点归一化 (0..1) 写入 objects[].keypoints，且 JSON 往返保真
// 5) 真实 Pipeline + loop 视频：sink → EventBus → 事件携带关键点
#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdlib>
#include <cstdio>
#include <fstream>
#include <set>
#include <string>
#include <thread>

#include "capability.hpp"
#include "config_adapter.hpp"
#include "event_bus.hpp"
#include "infer_group.hpp"
#include "inference_engine.hpp"
#include "nlohmann/json.hpp"
#include "pipeline.hpp"

using nlohmann::json;
using Catch::Approx;
using namespace modeldeploy::vision;

namespace {
const char* kPoseModel = "test_data/test_models/onnx/yolo11n/yolo11n-pose_nms.onnx";
const char* kPoseImage = "test_data/test_images/test_person.jpg";
const char* kPoseLoopVideo = "test_data/pose_person_loop.mp4";
// yolo11n-pose：COCO 17 点骨架
constexpr size_t kPoseKeypointCount = 17;

bool file_nonempty(const char* path) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    return f.good() && f.tellg() > 0;
}

bool ensure_loop_video(const char* img, const char* out) {
    if (file_nonempty(out)) return true;
    if (!file_nonempty(img)) return false;
    const std::string cmd = std::string("ffmpeg -y -loglevel error -loop 1 -i \"") + img +
                            "\" -vf \"scale=trunc(iw/2)*2:trunc(ih/2)*2\" -t 12 -r 8 "
                            "-pix_fmt yuv420p \"" + out + "\"";
    if (std::system(cmd.c_str()) != 0) return false;
    return file_nonempty(out);
}

ModelConfig make_pose_model() {
    ModelConfig m;
    m.name = "pose";
    m.type = "pose";
    m.backend = "ort";
    m.device = "cpu";
    m.path = kPoseModel;
    m.input_size = {640, 640};
    m.labels = {"person"};                    // yolo-pose 单类 person
    m.confidence_threshold = 0.25f;
    return m;
}
}  // namespace

// ── 类型名归一化：与云端场景目录声明（type="pose"）对齐，并接受常见别名 ──
TEST_CASE("normalize_model_type accepts pose aliases", "[agent][config]") {
    REQUIRE(normalize_model_type("pose") == "pose");
    REQUIRE(normalize_model_type("pose_estimation") == "pose");
    REQUIRE(normalize_model_type("keypoint") == "pose");
    REQUIRE(normalize_model_type("keypoints") == "pose");
    REQUIRE(normalize_model_type("keypoint_detection") == "pose");
    REQUIRE(normalize_model_type("human_pose") == "pose");
    REQUIRE(normalize_model_type("yolo_pose") == "pose");
}

TEST_CASE("detect_capabilities advertises pose family", "[agent][capability]") {
    auto c = detect_capabilities(8);
    std::set<std::string> fams;
    for (const auto& f : c["model_families"]) fams.insert(f.get<std::string>());
    REQUIRE(fams.count("pose") == 1);
}

TEST_CASE("ConfigAdapter parses pose task model", "[agent][config]") {
    json j = json::parse(R"({
      "task_id": 31,
      "camera": {"id": 1, "name": "c", "url": "test_data/test_video60.mp4"},
      "models": [{"name": "pose", "type": "pose", "backend": "ort", "device": "cpu",
                  "url": "test_data/test_models/onnx/yolo11n/yolo11n-pose_nms.onnx",
                  "input_size": [640, 640], "confidence_threshold": 0.25}],
      "events": {"transport": "http", "http": {"url": "http://x/cb", "token": "t"}}
    })");
    ConfigAdapter adapter;
    AdaptedTask t;
    std::string err;
    REQUIRE(adapter.from_json(j, &t, &err));
    REQUIRE(t.sdk.models.size() == 1);
    REQUIRE(t.sdk.models[0].type == "pose");
    REQUIRE(t.sdk.models[0].path == kPoseModel);
}

// ── 真机 smoke：模型输出人体框 + 每人 17 个关键点（像素坐标、落在帧内、分数 0..1）──
TEST_CASE("InferenceEngine pose yields 17 keypoints per person on real image",
          "[agent][model]") {
    if (!std::ifstream(kPoseModel).good() || !std::ifstream(kPoseImage).good())
        SKIP("pose model/image absent");
    InferenceEngine eng;
    REQUIRE(eng.load(make_pose_model()));
    auto img = ImageData::imread(kPoseImage);
    REQUIRE(!img.empty());
    InferResult r;
    REQUIRE(eng.infer(img, &r));
    REQUIRE(r.type == "pose");
    REQUIRE_FALSE(r.boxes.empty());

    size_t persons_with_kps = 0;
    for (const auto& b : r.boxes) {
        REQUIRE(b.w > 0.f);
        REQUIRE(b.h > 0.f);
        REQUIRE(b.score > 0.f);
        // 事件 label 必须非空（云端 object_present 依赖）
        REQUIRE_FALSE(b.label_name.empty());
        REQUIRE(b.keypoints.size() == kPoseKeypointCount);
        for (const auto& k : b.keypoints) {
            // 像素坐标（允许 letterbox 反变换后的轻微越界）
            REQUIRE(k[0] >= -static_cast<float>(img.width()) * 0.1f);
            REQUIRE(k[1] >= -static_cast<float>(img.height()) * 0.1f);
            REQUIRE(k[0] <= static_cast<float>(img.width()) * 1.1f);
            REQUIRE(k[1] <= static_cast<float>(img.height()) * 1.1f);
            REQUIRE(k[2] >= 0.f);
            REQUIRE(k[2] <= 1.f);
        }
        ++persons_with_kps;
    }
    WARN("pose smoke: persons=" << persons_with_kps << " keypoints/person="
                                << kPoseKeypointCount);
}

// pose 非 detection 模型，必须走 non_det 通道（detection 通道不受影响）
TEST_CASE("InferGroup routes pose through non-det path", "[agent][model]") {
    if (!std::ifstream(kPoseModel).good() || !std::ifstream(kPoseImage).good())
        SKIP("pose model/image absent");
    InferGroup g;
    REQUIRE(g.load_models({make_pose_model()}, nullptr));
    auto img = ImageData::imread(kPoseImage);
    REQUIRE(!img.empty());
    std::vector<std::pair<std::string, std::vector<DetectionResult>>> sdk_dets;
    std::vector<std::pair<std::string, InferResult>> non_det;
    g.run_models(img, &sdk_dets, &non_det);
    REQUIRE(sdk_dets.empty());
    REQUIRE(non_det.size() == 1);
    REQUIRE(non_det[0].second.type == "pose");
    REQUIRE_FALSE(non_det[0].second.boxes.empty());
}

// ── 事件契约（不经模型，锁定字段形态）：关键点归一化到 0..1 并写入 objects[].keypoints ──
TEST_CASE("pose keypoints normalized into event objects and JSON round-trip",
          "[agent][events]") {
    EventBus bus;
    std::vector<DetectionEvent> got;
    bus.set_sink([&](const DetectionEvent& e) { got.push_back(e); });
    EventMeta meta;
    meta.edge_code = "edge-01";
    meta.camera_id = 1;
    meta.task_id_num = 77;
    meta.alarm_interval_sec = 0;
    bus.register_task("t", meta);

    DetectionBox b;
    b.x = 100.f; b.y = 50.f; b.w = 200.f; b.h = 300.f;
    b.score = 0.9f; b.label_id = 0; b.label_name = "person";
    // 3 个关键点（像素）；含越界点验证钳制
    b.keypoints = {{100.f, 50.f, 0.9f}, {300.f, 200.f, 0.5f}, {700.f, 500.f, 0.2f}};
    bus.on_detections("t", {b}, 640, 480, 1.5);

    REQUIRE(got.size() == 1);
    auto j = got[0].to_json();
    REQUIRE(j["objects"].size() == 1);
    const auto& obj = j["objects"][0];
    REQUIRE(obj.contains("keypoints"));
    REQUIRE(obj["keypoints"].size() == 3);
    // 归一化 + 钳制到 0..1
    REQUIRE(obj["keypoints"][0][0].get<float>() == Approx(100.f / 640.f));
    REQUIRE(obj["keypoints"][0][1].get<float>() == Approx(50.f / 480.f));
    REQUIRE(obj["keypoints"][0][2].get<float>() == Approx(0.9f));
    REQUIRE(obj["keypoints"][1][0].get<float>() == Approx(300.f / 640.f));
    REQUIRE(obj["keypoints"][2][0].get<float>() <= 1.f);   // 700/640 被钳制
    REQUIRE(obj["keypoints"][2][1].get<float>() <= 1.f);   // 500/480 被钳制

    // JSON 往返：恢复后关键点数量/数值一致（归一化形态）
    auto e2 = DetectionEvent::from_json(j);
    REQUIRE(e2.detections.size() == 1);
    REQUIRE(e2.detections[0].keypoints.size() == 3);
    REQUIRE(e2.detections[0].keypoints[0][0] == Approx(100.f / 640.f));
    REQUIRE(e2.detections[0].keypoints[2][1] == Approx(1.f));
}

// 无关键点的普通检测事件不应新增 keypoints 字段（契约向后兼容）
TEST_CASE("event without keypoints omits keypoints field", "[agent][events]") {
    EventBus bus;
    std::vector<DetectionEvent> got;
    bus.set_sink([&](const DetectionEvent& e) { got.push_back(e); });
    EventMeta meta;
    meta.task_id_num = 1;
    meta.alarm_interval_sec = 0;
    bus.register_task("t", meta);

    DetectionBox b;
    b.x = 0.f; b.y = 0.f; b.w = 10.f; b.h = 10.f;
    b.score = 0.5f; b.label_id = 0; b.label_name = "person";
    bus.on_detections("t", {b}, 640, 480, 1.0);

    REQUIRE(got.size() == 1);
    auto j = got[0].to_json();
    REQUIRE_FALSE(j["objects"][0].contains("keypoints"));
}

// ── 集成：真实 Pipeline 把 pose 框（含关键点）送入 sink → 事件归一化 ──
TEST_CASE("Pipeline pose delivers keypoints to event sink", "[agent][pipeline][integration]") {
    if (!ensure_loop_video(kPoseImage, kPoseLoopVideo)) SKIP("pose image/ffmpeg absent");

    TaskConfig cfg;
    cfg.id = "pose_sink_test";
    cfg.input_url = kPoseLoopVideo;
    cfg.enable_preview = false;
    cfg.models.push_back(make_pose_model());

    EventBus bus;
    std::vector<DetectionEvent> got;
    bus.set_sink([&](const DetectionEvent& e) { got.push_back(e); });
    EventMeta meta;
    meta.edge_code = "edge-01";
    meta.camera_id = 1;
    meta.task_id_num = 88;
    meta.alarm_interval_sec = 0;
    bus.register_task(cfg.id, meta);

    Pipeline pipe(cfg);
    pipe.set_detection_sink([&](const std::vector<DetectionBox>& boxes, int w, int h, double lat) {
        bus.on_detections(cfg.id, boxes, w, h, lat);
    });
    REQUIRE(pipe.start());
    for (int i = 0; i < 400 && got.empty(); ++i)
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
    pipe.stop();

    REQUIRE_FALSE(got.empty());

    bool found = false;
    size_t max_persons = 0;
    for (const auto& e : got) {
        const auto j = e.to_json();
        size_t persons = 0;
        for (const auto& obj : j["objects"]) {
            if (!obj.contains("keypoints")) continue;
            REQUIRE(obj["keypoints"].size() == kPoseKeypointCount);
            REQUIRE_FALSE(obj.value("label", "").empty());
            for (const auto& k : obj["keypoints"]) {
                REQUIRE(k.size() == 3);
                REQUIRE(k[0].get<float>() >= 0.f);
                REQUIRE(k[0].get<float>() <= 1.f);
                REQUIRE(k[1].get<float>() >= 0.f);
                REQUIRE(k[1].get<float>() <= 1.f);
            }
            ++persons;
            found = true;
        }
        max_persons = std::max(max_persons, persons);
    }
    REQUIRE(found);
    WARN("pose pipeline smoke: events=" << got.size() << " max_persons=" << max_persons
                                        << " keypoints/person=" << kPoseKeypointCount);
}
