#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <atomic>
#include <chrono>
#include <cstdlib>
#include <fstream>
#include <string>
#include <thread>
#include <vector>

#include "event_bus.hpp"
#include "inference_engine.hpp"
#include "nlohmann/json.hpp"
#include "pipeline.hpp"

using nlohmann::json;
using Catch::Approx;
using namespace modeldeploy::vision;

namespace {
const char* kClsModel = "test_data/test_models/onnx/yolo11n/yolo11n-cls.onnx";
const char* kClsImage = "test_data/test_images/test_person.jpg";
const char* kClsLoopVideo = "test_data/cls_person_loop.mp4";

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

ModelConfig make_cls_model() {
    ModelConfig m;
    m.name = "cls";
    m.type = "classification";
    m.backend = "ort";
    m.device = "cpu";
    m.path = kClsModel;
    // ImageNet-1000 分类模型：给出足够长的标签表，label 形如 class<id>（非数字回退）
    m.labels.reserve(1000);
    for (int i = 0; i < 1000; ++i) m.labels.push_back("class" + std::to_string(i));
    m.confidence_threshold = 0.25f;
    return m;
}

// 复刻云端求值器 `_matches_label`（service.py）的 object_present/zone_enter 标签匹配语义。
bool cloud_matches_label(const json& obj, const json& leaf) {
    if (leaf.contains("labels") && leaf["labels"].is_array() && !leaf["labels"].empty()) {
        const std::string label = obj.value("label", "");
        for (const auto& l : leaf["labels"])
            if (l.is_string() && l.get<std::string>() == label) return true;
        return false;
    }
    if (!leaf.contains("label") || leaf["label"].is_null()) return true;
    return obj.value("label", "") == leaf["label"].get<std::string>();
}
}  // namespace

// ── 引擎契约：分类结果带整帧框 + top-1 label + 类别分数 attributes ──
TEST_CASE("InferenceEngine classification yields full-frame labelled box", "[agent][model]") {
    if (!std::ifstream(kClsModel).good() || !std::ifstream(kClsImage).good())
        SKIP("classification model/image absent");
    InferenceEngine eng;
    REQUIRE(eng.load(make_cls_model()));
    auto img = ImageData::imread(kClsImage);
    REQUIRE(!img.empty());
    InferResult r;
    REQUIRE(eng.infer(img, &r));
    REQUIRE(r.type == "classification");
    REQUIRE_FALSE(r.boxes.empty());
    const auto& b = r.boxes[0];
    // 整帧框（像素坐标）：事件归一化后即 0,0,1,1
    REQUIRE(b.x == 0.f);
    REQUIRE(b.y == 0.f);
    REQUIRE(b.w == static_cast<float>(img.width()));
    REQUIRE(b.h == static_cast<float>(img.height()));
    REQUIRE(b.score > 0.f);
    REQUIRE_FALSE(b.label_name.empty());
    REQUIRE(b.label_name == "class" + std::to_string(b.label_id));
    REQUIRE(b.attributes.size() == 1);
    REQUIRE(b.attributes[0].first == b.label_name);
}

// ── 事件契约（不经模型，锁定字段形态）：整帧框归一化为 0,0,1,1 且 label 落地 ──
TEST_CASE("classification full-frame box reaches event objects with label", "[agent][events]") {
    EventBus bus;
    std::vector<DetectionEvent> got;
    bus.set_sink([&](const DetectionEvent& e) { got.push_back(e); });
    EventMeta meta;
    meta.edge_code = "edge-01";
    meta.camera_id = 1;
    meta.task_id_num = 99;
    meta.alarm_interval_sec = 0;
    bus.register_task("t", meta);

    DetectionBox b;
    b.x = 0.f; b.y = 0.f; b.w = 640.f; b.h = 480.f;
    b.score = 0.87f; b.label_id = 3; b.label_name = "no_mask";
    b.attributes.emplace_back("no_mask", 0.87f);
    bus.on_detections("t", {b}, 640, 480, 2.0);

    REQUIRE(got.size() == 1);
    auto j = got[0].to_json();
    REQUIRE(j["objects"].size() == 1);
    const auto& obj = j["objects"][0];
    REQUIRE(obj["label"] == "no_mask");
    REQUIRE(obj["bbox"]["x"].get<float>() == Approx(0.f));
    REQUIRE(obj["bbox"]["y"].get<float>() == Approx(0.f));
    REQUIRE(obj["bbox"]["width"].get<float>() == Approx(1.f));
    REQUIRE(obj["bbox"]["height"].get<float>() == Approx(1.f));
    REQUIRE(obj["attributes"]["no_mask"].get<float>() == Approx(0.87f));
}

// ── 端到端：纯分类模型驱动真实 Pipeline → EventBus → 云端 object_present 叶子可命中 ──
TEST_CASE("classification-only model produces cloud-matchable event",
          "[agent][pipeline][integration]") {
    if (!ensure_loop_video(kClsImage, kClsLoopVideo)) SKIP("classification image/ffmpeg absent");

    TaskConfig cfg;
    cfg.id = "cls_event_test";
    cfg.input_url = kClsLoopVideo;
    cfg.enable_preview = false;
    cfg.models.push_back(make_cls_model());

    EventBus bus;
    std::vector<DetectionEvent> got;
    bus.set_sink([&](const DetectionEvent& e) { got.push_back(e); });
    EventMeta meta;
    meta.edge_code = "edge-01";
    meta.camera_id = 1;
    meta.task_id_num = 99;
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
    auto j = got[0].to_json();
    REQUIRE(j["objects"].size() >= 1);
    const auto& obj = j["objects"][0];
    REQUIRE(obj["bbox"]["x"].get<float>() == Approx(0.f));
    REQUIRE(obj["bbox"]["width"].get<float>() == Approx(1.f));
    const std::string label = obj["label"].get<std::string>();
    REQUIRE_FALSE(label.empty());

    // SCENE_CLS / DEFECT_CLS 默认规则 object_present（无 label）→ 命中
    REQUIRE(cloud_matches_label(obj, json{{"subject", "object_present"}}));
    // NO_MASK 形态 object_present（label=模型输出类名）→ 命中；无关键命中
    REQUIRE(cloud_matches_label(obj, json{{"subject", "object_present"}, {"label", label}}));
    REQUIRE_FALSE(cloud_matches_label(
        obj, json{{"subject", "object_present"}, {"label", "definitely_not_a_class"}}));
}
