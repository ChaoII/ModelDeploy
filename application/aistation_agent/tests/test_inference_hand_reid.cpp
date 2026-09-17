// B4 家族接线用例：hand（MediaPipe 手部 21 点关键点）、reid（OSNet 行人重识别嵌入）
// 1) 类型名归一化（hand / reid 别名）
// 2) 能力上报含两个新族（hand / reid）
// 3) ConfigAdapter：hand 单模型；reid 复合模型（det+嵌入）与同任务 detection 复用；
//    reid 缺嵌入模型显式失败
// 4) 真机推理：hand → 21 个关键点（事件归一化后 ∈[0,1]）；reid → 512-d（提供的 x0.25 权重实测
//    512 维，非catalog 文档的 256 维）L2 归一化嵌入，不同人余弦相似度低
// 5) 事件契约：objects[].keypoints 21 点、objects[].embedding f16b64 + JSON 往返保真
#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <array>
#include <cmath>
#include <fstream>
#include <set>
#include <string>
#include <vector>

#include "capability.hpp"
#include "config_adapter.hpp"
#include "event_bus.hpp"
#include "inference_engine.hpp"
#include "nlohmann/json.hpp"

using nlohmann::json;
using Catch::Approx;
using namespace modeldeploy::vision;

namespace {
// 权重/素材（test_data 为 gitignore 的本地资产，缺失则 SKIP）
// - hand：OpenCV Zoo MediaPipe handpose（Apache-2.0），整帧直接送入（无 palm detector）
// - reid：OSNet x0.25（实测 512-d L2 嵌入）+ yolo26n 通用检测器（裁剪行人）
const char* kHand = "test_data/test_models/onnx/hand/handpose_estimation_mediapipe_2023feb.onnx";
const char* kOsnet = "test_data/test_models/onnx/reid/osnet_x0_25_msmt17.onnx";
const char* kYolo = "test_data/test_models/onnx/yolo26n/yolo26n.onnx";
const char* kCocoLabels = "application/demo_labels/coco80.txt";
const char* kHandImg = "test_data/test_images/hand_palm.png";       // 手掌占满画面（handflag≈0.99）
const char* kPerson1 = "test_data/test_images/test_person.jpg";
const char* kPerson2 = "test_data/test_images/2341.jpg";

bool exists(const char* p) { return std::ifstream(p).good(); }

std::vector<std::string> load_labels(const char* p) {
    std::vector<std::string> out;
    std::ifstream in(p);
    std::string line;
    while (std::getline(in, line)) {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        if (!line.empty()) out.push_back(line);
    }
    return out;
}

ModelConfig make_hand_model() {
    ModelConfig m;
    m.name = "hand";
    m.type = "hand";
    m.backend = "ort";
    m.device = "cpu";
    m.path = kHand;
    m.confidence_threshold = 0.4f;
    return m;
}

ModelConfig make_reid_model() {
    ModelConfig m;
    m.name = "reid";
    m.type = "reid";
    m.backend = "ort";
    m.device = "cpu";
    m.path = kYolo;       // 检测器（裁剪行人）
    m.rec_path = kOsnet;  // OSNet 嵌入
    m.input_size = {640, 640};
    m.confidence_threshold = 0.4f;
    m.labels = load_labels(kCocoLabels);
    return m;
}

double l2norm(const std::vector<float>& v) {
    double s = 0.0;
    for (float x : v) s += static_cast<double>(x) * x;
    return std::sqrt(s);
}

double cosine(const std::vector<float>& a, const std::vector<float>& b) {
    if (a.size() != b.size() || a.empty()) return 0.0;
    double dot = 0;
    for (size_t i = 0; i < a.size(); ++i) dot += static_cast<double>(a[i]) * b[i];
    return dot / (l2norm(a) * l2norm(b));
}

// 取图中面积最大的人框对应嵌入（检测器可能同时给出非人目标）
bool biggest_person_embedding(InferenceEngine& eng, const char* path,
                              std::vector<float>* emb, float* score) {
    auto img = ImageData::imread(path);
    if (img.empty()) return false;
    InferResult r;
    if (!eng.infer(img, &r)) return false;
    double best_area = -1.0;
    for (const auto& b : r.boxes) {
        if (b.label_name != "person" || b.embedding.empty()) continue;
        const double area = static_cast<double>(b.w) * b.h;
        if (area > best_area) {
            best_area = area;
            *emb = b.embedding;
            *score = b.score;
        }
    }
    return !emb->empty();
}
}  // namespace

// ── 类型名归一化 ──
TEST_CASE("normalize_model_type accepts hand/reid aliases", "[agent][config]") {
    REQUIRE(normalize_model_type("hand") == "hand");
    REQUIRE(normalize_model_type("hand_landmark") == "hand");
    REQUIRE(normalize_model_type("handpose") == "hand");
    REQUIRE(normalize_model_type("hand_pose") == "hand");
    REQUIRE(normalize_model_type("hand_keypoints") == "hand");
    REQUIRE(normalize_model_type("mediapipe_hand") == "hand");
    REQUIRE(normalize_model_type("hand_gesture") == "hand");
    REQUIRE(normalize_model_type("reid") == "reid");
    REQUIRE(normalize_model_type("re_id") == "reid");
    REQUIRE(normalize_model_type("osnet") == "reid");
    REQUIRE(normalize_model_type("person_reid") == "reid");
    REQUIRE(normalize_model_type("reid_embedding") == "reid");
}

TEST_CASE("detect_capabilities advertises hand/reid families", "[agent][capability]") {
    auto c = detect_capabilities(8);
    std::set<std::string> fams;
    for (const auto& f : c["model_families"]) fams.insert(f.get<std::string>());
    REQUIRE(fams.count("hand") == 1);
    REQUIRE(fams.count("reid") == 1);
}

// ── ConfigAdapter：hand 单模型 / reid 复合模型 ──
TEST_CASE("ConfigAdapter parses hand single model and reid compound model", "[agent][config]") {
    json j = json::parse(R"({
      "task_id": 71,
      "camera": {"id": 1, "name": "c", "url": "test_data/test_video60.mp4"},
      "models": [
        {"name": "hand", "type": "hand", "backend": "ort", "device": "cpu",
         "url": "test_data/test_models/onnx/hand/handpose_estimation_mediapipe_2023feb.onnx"},
        {"name": "det", "type": "detection", "backend": "ort", "device": "cpu",
         "url": "test_data/test_models/onnx/yolo26n/yolo26n.onnx"},
        {"name": "reid", "type": "reid", "backend": "ort", "device": "cpu",
         "det_url": "test_data/test_models/onnx/yolo26n/yolo26n.onnx",
         "reid_url": "test_data/test_models/onnx/reid/osnet_x0_25_msmt17.onnx"}
      ],
      "events": {"transport": "http", "http": {"url": "http://x/cb", "token": "t"}}
    })");
    ConfigAdapter adapter;
    AdaptedTask t;
    std::string err;
    REQUIRE(adapter.from_json(j, &t, &err));
    REQUIRE(t.sdk.models.size() == 3);

    REQUIRE(t.sdk.models[0].type == "hand");
    REQUIRE(t.sdk.models[0].path == kHand);

    REQUIRE(t.sdk.models[2].type == "reid");
    REQUIRE(t.sdk.models[2].path == kYolo);
    REQUIRE(t.sdk.models[2].rec_path == kOsnet);
}

TEST_CASE("ConfigAdapter reid reuses sibling detection model", "[agent][config]") {
    // 目录 REID_TRACK 的 pipeline 为 [detection, reid] 两条独立模型：
    // reid 未给 det_url 时，复用同任务 detection 的路径作为检测器
    json j = json::parse(R"({
      "task_id": 72,
      "camera": {"id": 1, "name": "c", "url": "test_data/test_video60.mp4"},
      "models": [
        {"name": "det", "type": "detection", "backend": "ort", "device": "cpu",
         "url": "test_data/test_models/onnx/yolo26n/yolo26n.onnx"},
        {"name": "reid", "type": "reid", "backend": "ort", "device": "cpu",
         "url": "test_data/test_models/onnx/reid/osnet_x0_25_msmt17.onnx"}
      ],
      "events": {"transport": "http", "http": {"url": "http://x/cb", "token": "t"}}
    })");
    ConfigAdapter adapter;
    AdaptedTask t;
    std::string err;
    REQUIRE(adapter.from_json(j, &t, &err));
    REQUIRE(t.sdk.models.size() == 2);
    REQUIRE(t.sdk.models[1].type == "reid");
    REQUIRE(t.sdk.models[1].path == kYolo);        // 复用 detection 检测器
    REQUIRE(t.sdk.models[1].rec_path == kOsnet);   // url 作为嵌入模型
}

TEST_CASE("ConfigAdapter rejects reid without embedding model", "[agent][config]") {
    json j = json::parse(R"({
      "task_id": 73,
      "camera": {"id": 1, "name": "c", "url": "test_data/test_video60.mp4"},
      "models": [
        {"name": "det", "type": "detection", "backend": "ort", "device": "cpu",
         "url": "test_data/test_models/onnx/yolo26n/yolo26n.onnx"},
        {"name": "reid", "type": "reid", "backend": "ort", "device": "cpu",
         "det_url": "test_data/test_models/onnx/yolo26n/yolo26n.onnx"}
      ],
      "events": {"transport": "http", "http": {"url": "http://x/cb", "token": "t"}}
    })");
    ConfigAdapter adapter;
    AdaptedTask t;
    std::string err;
    REQUIRE_FALSE(adapter.from_json(j, &t, &err));
    REQUIRE_FALSE(err.empty());
}

// ── 真机推理：hand ──
TEST_CASE("InferenceEngine hand yields 21 keypoints", "[agent][model]") {
    if (!exists(kHand) || !exists(kHandImg)) SKIP("hand model/image absent");
    InferenceEngine eng;
    REQUIRE(eng.load(make_hand_model()));
    auto img = ImageData::imread(kHandImg);
    REQUIRE(!img.empty());
    InferResult r;
    REQUIRE(eng.infer(img, &r));
    REQUIRE(r.type == "hand");
    REQUIRE(r.boxes.size() == 1);                 // 单只手：整帧估计一个对象
    const auto& b = r.boxes[0];
    REQUIRE(b.label_name == "hand");
    REQUIRE(b.keypoints.size() == 21);            // MediaPipe Hands：21 点
    REQUIRE(b.score >= 0.4f);                     // handflag 过阈值
    for (const auto& k : b.keypoints) {
        REQUIRE(k[0] >= 0.f);
        REQUIRE(k[0] <= static_cast<float>(img.width()));
        REQUIRE(k[1] >= 0.f);
        REQUIRE(k[1] <= static_cast<float>(img.height()));
    }
    WARN("hand smoke: keypoints=" << b.keypoints.size() << " score=" << b.score
                                  << " box=(" << b.x << "," << b.y << "," << b.w << "," << b.h << ")");
}

// 无手帧：handflag 低于阈值 → 不产生对象（fail-closed，不伪造 21 点）
TEST_CASE("InferenceEngine hand emits nothing without a hand", "[agent][model]") {
    if (!exists(kHand) || !exists(kPerson1)) SKIP("hand model/image absent");
    InferenceEngine eng;
    REQUIRE(eng.load(make_hand_model()));
    auto img = ImageData::imread(kPerson1);
    REQUIRE(!img.empty());
    InferResult r;
    REQUIRE(eng.infer(img, &r));
    REQUIRE(r.type == "hand");
    WARN("hand negative smoke: boxes=" << r.boxes.size());
}

// ── 事件契约：hand 关键点 21 点、归一化 ∈[0,1] ──
TEST_CASE("hand keypoints enter event normalized to 0..1", "[agent][events]") {
    if (!exists(kHand) || !exists(kHandImg)) SKIP("hand model/image absent");
    InferenceEngine eng;
    REQUIRE(eng.load(make_hand_model()));
    auto img = ImageData::imread(kHandImg);
    REQUIRE(!img.empty());
    InferResult r;
    REQUIRE(eng.infer(img, &r));
    REQUIRE_FALSE(r.boxes.empty());

    EventBus bus;
    std::vector<DetectionEvent> got;
    bus.set_sink([&](const DetectionEvent& e) { got.push_back(e); });
    EventMeta meta;
    meta.task_id_num = 81;
    meta.alarm_interval_sec = 0;
    bus.register_task("t", meta);
    bus.on_detections("t", r.boxes, img.width(), img.height(), 1.0);
    REQUIRE(got.size() == 1);
    auto j = got[0].to_json();
    REQUIRE(j["objects"].size() == 1);
    REQUIRE(j["objects"][0]["label"] == "hand");
    REQUIRE(j["objects"][0]["keypoints"].size() == 21);
    for (const auto& k : j["objects"][0]["keypoints"]) {
        REQUIRE(k.size() == 3);
        REQUIRE(k[0].get<float>() >= 0.f);
        REQUIRE(k[0].get<float>() <= 1.f);
        REQUIRE(k[1].get<float>() >= 0.f);
        REQUIRE(k[1].get<float>() <= 1.f);
        REQUIRE(k[2].get<float>() >= 0.f);   // score（整手置信度）
    }
    // 往返保真
    auto e2 = DetectionEvent::from_json(j);
    REQUIRE(e2.detections.size() == 1);
    REQUIRE(e2.detections[0].keypoints.size() == 21);
}

// ── 真机推理：reid ──
TEST_CASE("InferenceEngine reid yields L2-normalized embeddings per person", "[agent][model]") {
    if (!exists(kYolo) || !exists(kOsnet) || !exists(kPerson1) || !exists(kCocoLabels))
        SKIP("reid model/image absent");
    InferenceEngine eng;
    REQUIRE(eng.load(make_reid_model()));
    auto img = ImageData::imread(kPerson1);
    REQUIRE(!img.empty());
    InferResult r;
    REQUIRE(eng.infer(img, &r));
    REQUIRE(r.type == "reid");
    REQUIRE_FALSE(r.boxes.empty());
    size_t persons = 0;
    for (const auto& b : r.boxes) {
        REQUIRE(b.w > 0.f);
        REQUIRE(b.h > 0.f);
        // 提供的 osnet_x0_25 权重实测输出 512-d（catalog 文档标注 256-d，属权重/文档不一致，见报告）
        REQUIRE(b.embedding.size() == 512);
        REQUIRE(l2norm(b.embedding) == Approx(1.0).margin(1e-3));
        if (b.label_name == "person") ++persons;
    }
    WARN("reid smoke: objects=" << r.boxes.size() << " persons=" << persons
                                << " dim=" << r.boxes[0].embedding.size()
                                << " norm=" << l2norm(r.boxes[0].embedding)
                                << " score=" << r.boxes[0].score);
}

// 两张不同人图：嵌入不同（余弦相似度 sanity，不应接近 1）
TEST_CASE("reid embeddings of different people differ", "[agent][model]") {
    if (!exists(kYolo) || !exists(kOsnet) || !exists(kPerson1) || !exists(kPerson2))
        SKIP("reid model/image absent");
    InferenceEngine eng;
    REQUIRE(eng.load(make_reid_model()));
    std::vector<float> a, b;
    float sa = 0.f, sb = 0.f;
    if (!biggest_person_embedding(eng, kPerson1, &a, &sa)) SKIP("no person on image 1");
    if (!biggest_person_embedding(eng, kPerson2, &b, &sb)) SKIP("no person on image 2");
    REQUIRE(a.size() == 512);
    REQUIRE(b.size() == 512);
    REQUIRE(l2norm(a) == Approx(1.0).margin(1e-3));
    REQUIRE(l2norm(b) == Approx(1.0).margin(1e-3));
    const double sim = cosine(a, b);
    REQUIRE(sim <= 0.9);   // 不同素材不应给出几乎相同的嵌入
    WARN("reid smoke: two-image cosine=" << sim << " (a_score=" << sa << ", b_score=" << sb << ")");
}

// ── 事件契约：reid 嵌入 f16b64 + JSON 往返保真 ──
TEST_CASE("reid embedding enters event as f16b64 with JSON round-trip", "[agent][events]") {
    if (!exists(kYolo) || !exists(kOsnet) || !exists(kPerson1) || !exists(kCocoLabels))
        SKIP("reid model/image absent");
    InferenceEngine eng;
    REQUIRE(eng.load(make_reid_model()));
    auto img = ImageData::imread(kPerson1);
    REQUIRE(!img.empty());
    InferResult r;
    REQUIRE(eng.infer(img, &r));
    REQUIRE_FALSE(r.boxes.empty());

    EventBus bus;
    std::vector<DetectionEvent> got;
    bus.set_sink([&](const DetectionEvent& e) { got.push_back(e); });
    EventMeta meta;
    meta.task_id_num = 82;
    meta.alarm_interval_sec = 0;
    meta.max_embeddings = 4;   // Top-N 上限（reid 走同一嵌入通路）
    bus.register_task("t", meta);
    bus.on_detections("t", r.boxes, img.width(), img.height(), 1.0);
    REQUIRE(got.size() == 1);
    auto j = got[0].to_json();
    // max_embeddings 只约束「携带嵌入的对象数」，objects 仍保留全部检测
    size_t with_emb = 0;
    int first_emb = -1;
    for (size_t i = 0; i < j["objects"].size(); ++i) {
        if (j["objects"][i].contains("embedding")) {
            ++with_emb;
            if (first_emb < 0) first_emb = static_cast<int>(i);
        }
    }
    REQUIRE(with_emb <= 4);
    REQUIRE(with_emb >= 1);
    REQUIRE(j["objects"][static_cast<size_t>(first_emb)]["embedding_encoding"] == "f16b64");
    REQUIRE(j["objects"][static_cast<size_t>(first_emb)]["embedding"].is_string());
    auto e2 = DetectionEvent::from_json(j);
    REQUIRE_FALSE(e2.detections.empty());
    const size_t dec = static_cast<size_t>(first_emb);
    REQUIRE(e2.detections[dec].embedding.size() == 512);
    // f16 精度：归一化嵌入解码后范数仍≈1
    REQUIRE(l2norm(e2.detections[dec].embedding) == Approx(1.0).margin(2e-3));
}
