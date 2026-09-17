// B3 家族接线用例：sam（FastSAM-s 交互式分割）、face_rec（人脸嵌入）
// 1) 类型名归一化（sam / face_rec 别名）
// 2) 能力上报含两个新族（sam / face_rec）
// 3) ConfigAdapter：sam 提示点解析（模型级/任务级）、face_rec 复合模型（det+嵌入）与
//    face_detection 同任务复用
// 4) 真机推理：sam → 提示点选中掩码的单个 segment（label="segment"，bbox 归一化后 ∈[0,1]）
//    face_rec → 每人一个 512-d 嵌入（已 L2 归一化，|v|≈1）
// 5) 事件契约：objects[].embedding 写入 + JSON 往返保真；无嵌入时不写该字段
#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <array>
#include <cmath>
#include <fstream>
#include <map>
#include <set>
#include <string>
#include <vector>

#include "capability.hpp"
#include "config_adapter.hpp"
#include "event_bus.hpp"
#include "inference_engine.hpp"
#include "nlohmann/json.hpp"

// SDK 直用：在真机 smoke 中定位一个「确定落在掩码内」的提示点（验证提示点→掩码映射）
#include "csrc/runtime/runtime_option.h"
#include "csrc/vision/sam/fastsam.h"

using nlohmann::json;
using Catch::Approx;
using namespace modeldeploy::vision;

namespace {
const char* kFastSam = "test_data/test_models/onnx/FastSAM-s.onnx";
const char* kScrfd = "test_data/test_models/onnx/seetaface/scrfd_2.5g_bnkps_shape640x640.onnx";
const char* kW600k = "test_data/test_models/onnx/insightface/buffalo_l/w600k_r50.onnx";
const char* kSamImg = "test_data/test_images/bus.jpg";
const char* kFace1 = "test_data/test_images/test_face_detection4.jpg";
const char* kFace2 = "test_data/test_images/test_face_detection5.jpg";

bool exists(const char* p) { return std::ifstream(p).good(); }

ModelConfig make_sam_model(const std::vector<float>& prompt = {}) {
    ModelConfig m;
    m.name = "sam";
    m.type = "sam";
    m.backend = "ort";
    m.device = "cpu";
    m.path = kFastSam;
    m.input_size = {640, 640};
    m.confidence_threshold = 0.3f;
    m.prompt_point = prompt;
    return m;
}

ModelConfig make_face_rec_model() {
    ModelConfig m;
    m.name = "face_rec";
    m.type = "face_rec";
    m.backend = "ort";
    m.device = "cpu";
    m.path = kScrfd;      // det
    m.rec_path = kW600k;  // 嵌入（112→512-d）
    m.input_size = {640, 640};
    m.confidence_threshold = 0.3f;
    return m;
}

double l2norm(const std::vector<float>& v) {
    double s = 0.0;
    for (float x : v) s += static_cast<double>(x) * x;
    return std::sqrt(s);
}

double cosine(const std::vector<float>& a, const std::vector<float>& b) {
    if (a.size() != b.size() || a.empty()) return 0.0;
    double dot = 0, na = 0, nb = 0;
    for (size_t i = 0; i < a.size(); ++i) {
        dot += static_cast<double>(a[i]) * b[i];
        na += static_cast<double>(a[i]) * a[i];
        nb += static_cast<double>(b[i]) * b[i];
    }
    if (na <= 0 || nb <= 0) return 0.0;
    return dot / (std::sqrt(na) * std::sqrt(nb));
}

// 用 SDK FastSam 找一个确定落在某掩码内的像素点（返回归一化坐标）：
// 取第一个实例掩码的「非边界前景像素」，该点必然命中掩码 → bbox 必然包含它。
bool find_mask_point(const char* img_path, float* nx, float* ny) {
    modeldeploy::RuntimeOption opt;
    opt.use_cpu();
    seg::FastSam sam(kFastSam, opt);
    if (!sam.is_initialized()) return false;
    sam.get_preprocessor().set_size({640, 640});
    sam.get_postprocessor().set_conf_threshold(0.3f);
    auto img = ImageData::imread(img_path);
    if (img.empty()) return false;
    std::vector<InstanceSegResult> segs;
    if (!sam.predict(img, &segs)) return false;
    for (const auto& s : segs) {
        if (s.mask.shape.size() < 2) continue;
        const int h = static_cast<int>(s.mask.shape[0]);
        const int w = static_cast<int>(s.mask.shape[1]);
        for (int y = 1; y + 1 < h; ++y) {
            for (int x = 1; x + 1 < w; ++x) {
                if (!s.mask.buffer[static_cast<size_t>(y) * w + x]) continue;
                *nx = static_cast<float>(s.box.x + x) / static_cast<float>(img.width());
                *ny = static_cast<float>(s.box.y + y) / static_cast<float>(img.height());
                return true;
            }
        }
    }
    return false;
}
}  // namespace

// ── 类型名归一化 ──
TEST_CASE("normalize_model_type accepts sam/face_rec aliases", "[agent][config]") {
    REQUIRE(normalize_model_type("sam") == "sam");
    REQUIRE(normalize_model_type("fastsam") == "sam");
    REQUIRE(normalize_model_type("fast_sam") == "sam");
    REQUIRE(normalize_model_type("sam_seg") == "sam");
    REQUIRE(normalize_model_type("segment_anything") == "sam");
    REQUIRE(normalize_model_type("face_rec") == "face_rec");
    REQUIRE(normalize_model_type("face_recognition") == "face_rec");
    REQUIRE(normalize_model_type("face_embedding") == "face_rec");
    REQUIRE(normalize_model_type("arcface") == "face_rec");
}

TEST_CASE("detect_capabilities advertises sam/face_rec families", "[agent][capability]") {
    auto c = detect_capabilities(8);
    std::set<std::string> fams;
    for (const auto& f : c["model_families"]) fams.insert(f.get<std::string>());
    REQUIRE(fams.count("sam") == 1);
    REQUIRE(fams.count("face_rec") == 1);
}

// ── ConfigAdapter：提示点解析 + face_rec 复合模型 ──
TEST_CASE("ConfigAdapter parses sam prompt_point and face_rec compound models", "[agent][config]") {
    json j = json::parse(R"({
      "task_id": 51,
      "camera": {"id": 1, "name": "c", "url": "test_data/test_video60.mp4"},
      "prompt_point": [0.25, 0.75],
      "models": [
        {"name": "sam", "type": "sam", "backend": "ort", "device": "cpu",
         "url": "test_data/test_models/onnx/FastSAM-s.onnx"},
        {"name": "fr", "type": "face_rec", "backend": "ort", "device": "cpu",
         "det_url": "test_data/test_models/onnx/seetaface/scrfd_2.5g_bnkps_shape640x640.onnx",
         "rec_url": "test_data/test_models/onnx/insightface/buffalo_l/w600k_r50.onnx"}
      ],
      "events": {"transport": "http", "http": {"url": "http://x/cb", "token": "t"}}
    })");
    ConfigAdapter adapter;
    AdaptedTask t;
    std::string err;
    REQUIRE(adapter.from_json(j, &t, &err));
    REQUIRE(t.sdk.models.size() == 2);

    // sam：任务级 prompt_point 回退（模型条目未带提示点）
    REQUIRE(t.sdk.models[0].type == "sam");
    REQUIRE(t.sdk.models[0].path == kFastSam);
    REQUIRE(t.sdk.models[0].prompt_point.size() == 2);
    REQUIRE(t.sdk.models[0].prompt_point[0] == Approx(0.25f));
    REQUIRE(t.sdk.models[0].prompt_point[1] == Approx(0.75f));

    // face_rec：det + 嵌入两路径
    REQUIRE(t.sdk.models[1].type == "face_rec");
    REQUIRE(t.sdk.models[1].path == kScrfd);
    REQUIRE(t.sdk.models[1].rec_path == kW600k);
}

TEST_CASE("ConfigAdapter face_rec reuses sibling face_detection model", "[agent][config]") {
    // 目录 FACE_REC 的 pipeline 为 [face_detection, face_rec] 两条独立模型：
    // face_rec 未给 det_url 时，复用同任务 face_detection 的路径作为检测器
    json j = json::parse(R"({
      "task_id": 52,
      "camera": {"id": 1, "name": "c", "url": "test_data/test_video60.mp4"},
      "models": [
        {"name": "fd", "type": "face_detection", "backend": "ort", "device": "cpu",
         "url": "test_data/test_models/onnx/seetaface/scrfd_2.5g_bnkps_shape640x640.onnx"},
        {"name": "fr", "type": "face_rec", "backend": "ort", "device": "cpu",
         "url": "test_data/test_models/onnx/insightface/buffalo_l/w600k_r50.onnx"}
      ],
      "events": {"transport": "http", "http": {"url": "http://x/cb", "token": "t"}}
    })");
    ConfigAdapter adapter;
    AdaptedTask t;
    std::string err;
    REQUIRE(adapter.from_json(j, &t, &err));
    REQUIRE(t.sdk.models.size() == 2);
    REQUIRE(t.sdk.models[1].type == "face_rec");
    REQUIRE(t.sdk.models[1].path == kScrfd);     // 复用 face_detection 检测器
    REQUIRE(t.sdk.models[1].rec_path == kW600k); // url 作为嵌入模型
}

TEST_CASE("ConfigAdapter parses model-level prompt_point object", "[agent][config]") {
    json j = json::parse(R"({
      "task_id": 53,
      "camera": {"id": 1, "name": "c", "url": "test_data/test_video60.mp4"},
      "models": [
        {"name": "sam", "type": "sam", "backend": "ort", "device": "cpu",
         "url": "test_data/test_models/onnx/FastSAM-s.onnx",
         "prompt_point": {"x": 0.4, "y": 0.6}}
      ],
      "events": {"transport": "http", "http": {"url": "http://x/cb", "token": "t"}}
    })");
    ConfigAdapter adapter;
    AdaptedTask t;
    std::string err;
    REQUIRE(adapter.from_json(j, &t, &err));
    REQUIRE(t.sdk.models.size() == 1);
    REQUIRE(t.sdk.models[0].prompt_point.size() == 2);
    REQUIRE(t.sdk.models[0].prompt_point[0] == Approx(0.4f));
    REQUIRE(t.sdk.models[0].prompt_point[1] == Approx(0.6f));
}

// face_rec 缺嵌入模型必须显式失败（不静默）
TEST_CASE("ConfigAdapter rejects face_rec without embedding model", "[agent][config]") {
    json j = json::parse(R"({
      "task_id": 54,
      "camera": {"id": 1, "name": "c", "url": "test_data/test_video60.mp4"},
      "models": [
        {"name": "fd", "type": "face_detection", "backend": "ort", "device": "cpu",
         "url": "test_data/test_models/onnx/seetaface/scrfd_2.5g_bnkps_shape640x640.onnx"},
        {"name": "fr", "type": "face_rec", "backend": "ort", "device": "cpu"}
      ],
      "events": {"transport": "http", "http": {"url": "http://x/cb", "token": "t"}}
    })");
    ConfigAdapter adapter;
    AdaptedTask t;
    std::string err;
    REQUIRE_FALSE(adapter.from_json(j, &t, &err));
    REQUIRE_FALSE(err.empty());
}

// ── 真机推理：sam（提示点选段） ──
TEST_CASE("InferenceEngine sam selects the prompted segment on real image", "[agent][model]") {
    if (!exists(kFastSam) || !exists(kSamImg)) SKIP("FastSAM model/image absent");
    float nx = 0.f, ny = 0.f;
    if (!find_mask_point(kSamImg, &nx, &ny)) SKIP("no mask pixel found on sample image");

    InferenceEngine eng;
    REQUIRE(eng.load(make_sam_model({nx, ny})));
    auto img = ImageData::imread(kSamImg);
    REQUIRE(!img.empty());
    InferResult r;
    REQUIRE(eng.infer(img, &r));
    REQUIRE(r.type == "sam");
    REQUIRE(r.boxes.size() == 1);              // 只输出选中的单个 segment
    const auto& b = r.boxes[0];
    REQUIRE(b.label_name == "segment");
    REQUIRE(b.w > 0.f);
    REQUIRE(b.h > 0.f);
    // 提示点命中的掩码 ⊂ bbox ⇒ 选中框必然包含提示点（映射正确性的可判定断言）
    const float px = nx * static_cast<float>(img.width());
    const float py = ny * static_cast<float>(img.height());
    REQUIRE(b.x <= px + 1.f);
    REQUIRE(b.y <= py + 1.f);
    REQUIRE(b.x + b.w >= px - 1.f);
    REQUIRE(b.y + b.h >= py - 1.f);
    WARN("sam smoke: prompt=(" << nx << "," << ny << ") bbox=(" << b.x << "," << b.y
                               << "," << b.w << "," << b.h << ") score=" << b.score);
}

// 无提示点 → 回退最高分掩码（仍产出单个 segment 对象）
TEST_CASE("InferenceEngine sam falls back to highest-score mask without prompt",
          "[agent][model]") {
    if (!exists(kFastSam) || !exists(kSamImg)) SKIP("FastSAM model/image absent");
    InferenceEngine eng;
    REQUIRE(eng.load(make_sam_model()));
    auto img = ImageData::imread(kSamImg);
    REQUIRE(!img.empty());
    InferResult r;
    REQUIRE(eng.infer(img, &r));
    REQUIRE(r.type == "sam");
    REQUIRE(r.boxes.size() == 1);
    REQUIRE(r.boxes[0].label_name == "segment");
    REQUIRE(r.boxes[0].score > 0.f);
    WARN("sam fallback smoke: score=" << r.boxes[0].score);
}

// 事件层：归一化 bbox（0..1）+ label="segment"
TEST_CASE("sam segment enters event with normalized bbox", "[agent][events]") {
    if (!exists(kFastSam) || !exists(kSamImg)) SKIP("FastSAM model/image absent");
    InferenceEngine eng;
    REQUIRE(eng.load(make_sam_model()));
    auto img = ImageData::imread(kSamImg);
    REQUIRE(!img.empty());
    InferResult r;
    REQUIRE(eng.infer(img, &r));
    REQUIRE_FALSE(r.boxes.empty());

    EventBus bus;
    std::vector<DetectionEvent> got;
    bus.set_sink([&](const DetectionEvent& e) { got.push_back(e); });
    EventMeta meta;
    meta.task_id_num = 61;
    meta.alarm_interval_sec = 0;
    bus.register_task("t", meta);
    bus.on_detections("t", r.boxes, img.width(), img.height(), 1.0);
    REQUIRE(got.size() == 1);
    auto j = got[0].to_json();
    REQUIRE(j["objects"].size() == 1);
    REQUIRE(j["objects"][0]["label"] == "segment");
    const auto& bb = j["objects"][0]["bbox"];
    REQUIRE(bb["x"].get<float>() >= 0.f);
    REQUIRE(bb["y"].get<float>() >= 0.f);
    REQUIRE(bb["x"].get<float>() + bb["width"].get<float>() <= Approx(1.01f));
    REQUIRE(bb["y"].get<float>() + bb["height"].get<float>() <= Approx(1.01f));
}

// ── 真机推理：face_rec（嵌入） ──
TEST_CASE("InferenceEngine face_rec yields L2-normalized embeddings per face", "[agent][model]") {
    if (!exists(kScrfd) || !exists(kW600k) || !exists(kFace1)) SKIP("face_rec model/image absent");
    InferenceEngine eng;
    REQUIRE(eng.load(make_face_rec_model()));
    auto img = ImageData::imread(kFace1);
    REQUIRE(!img.empty());
    InferResult r;
    REQUIRE(eng.infer(img, &r));
    REQUIRE(r.type == "face_rec");
    REQUIRE_FALSE(r.boxes.empty());   // 该素材 SCRFD 可检出人脸
    size_t faces = 0;
    for (const auto& b : r.boxes) {
        REQUIRE(b.label_name == "face");
        REQUIRE(b.w > 0.f);
        REQUIRE(b.h > 0.f);
        REQUIRE(b.embedding.size() == 512);                       // w600k_r50 → 512-d
        REQUIRE(l2norm(b.embedding) == Approx(1.0).margin(1e-3)); // 已 L2 归一化
        if (faces == 0) WARN("face_rec smoke: dim=" << b.embedding.size()
                                                    << " norm=" << l2norm(b.embedding)
                                                    << " score=" << b.score
                                                    << " json_bytes="
                                                    << json(b.embedding).dump().size());
        ++faces;
    }
    WARN("face_rec smoke: faces=" << faces);
}

// 两张不同人脸：嵌入不同（余弦相似度 sanity，不应接近 1）
TEST_CASE("face_rec embeddings of different faces differ", "[agent][model]") {
    if (!exists(kScrfd) || !exists(kW600k) || !exists(kFace1) || !exists(kFace2))
        SKIP("face_rec model/image absent");
    InferenceEngine eng;
    REQUIRE(eng.load(make_face_rec_model()));
    auto run_first = [&](const char* p, double* norm) -> std::vector<float> {
        auto img = ImageData::imread(p);
        if (img.empty()) return {};
        InferResult r;
        if (!eng.infer(img, &r) || r.boxes.empty()) return {};
        *norm = l2norm(r.boxes[0].embedding);
        return r.boxes[0].embedding;
    };
    double n1 = 0.0, n2 = 0.0;
    const auto a = run_first(kFace1, &n1);
    const auto b = run_first(kFace2, &n2);
    if (a.empty() || b.empty()) SKIP("face not detected on one of the sample images");
    REQUIRE(a.size() == 512);
    REQUIRE(b.size() == 512);
    REQUIRE(n1 == Approx(1.0).margin(1e-3));
    REQUIRE(n2 == Approx(1.0).margin(1e-3));
    const double sim = cosine(a, b);
    REQUIRE(sim <= 0.99);   // 不同素材不应给出几乎相同的嵌入
    WARN("face_rec smoke: two-image cosine=" << sim);
}

// ── 事件契约：objects[].embedding 写入 + JSON 往返保真 ──
TEST_CASE("face embedding enters event objects with JSON round-trip", "[agent][events]") {
    EventBus bus;
    std::vector<DetectionEvent> got;
    bus.set_sink([&](const DetectionEvent& e) { got.push_back(e); });
    EventMeta meta;
    meta.edge_code = "edge-01";
    meta.camera_id = 1;
    meta.task_id_num = 62;
    meta.alarm_interval_sec = 0;
    bus.register_task("t", meta);

    DetectionBox face;
    face.x = 100.f; face.y = 50.f; face.w = 200.f; face.h = 200.f;
    face.score = 0.9f; face.label_id = 0; face.label_name = "face";
    face.embedding = {0.6f, 0.8f};   // 简化向量（真实为 512-d）

    bus.on_detections("t", {face}, 640, 480, 1.0);
    REQUIRE(got.size() == 1);
    auto j = got[0].to_json();
    REQUIRE(j["objects"].size() == 1);
    REQUIRE(j["objects"][0].contains("embedding"));
    // 紧凑编码：base64(float16) + 显式编码标识，避免原始 float 数组膨胀上行体积
    REQUIRE(j["objects"][0]["embedding_encoding"] == "f16b64");
    REQUIRE(j["objects"][0]["embedding"].is_string());
    REQUIRE(j["schema_version"] == 2);

    auto e2 = DetectionEvent::from_json(j);
    REQUIRE(e2.detections.size() == 1);
    REQUIRE(e2.detections[0].embedding.size() == 2);
    REQUIRE(e2.detections[0].embedding[0] == Approx(0.6f).margin(1e-3));
    REQUIRE(e2.detections[0].embedding[1] == Approx(0.8f).margin(1e-3));
}

// 无嵌入的普通事件不新增 embedding 字段（契约向后兼容）
TEST_CASE("event without embedding omits embedding field", "[agent][events]") {
    EventBus bus;
    std::vector<DetectionEvent> got;
    bus.set_sink([&](const DetectionEvent& e) { got.push_back(e); });
    EventMeta meta;
    meta.task_id_num = 63;
    meta.alarm_interval_sec = 0;
    bus.register_task("t", meta);

    DetectionBox b;
    b.x = 0.f; b.y = 0.f; b.w = 10.f; b.h = 10.f;
    b.score = 0.5f; b.label_id = 0; b.label_name = "person";
    bus.on_detections("t", {b}, 640, 480, 1.0);

    REQUIRE(got.size() == 1);
    auto j = got[0].to_json();
    REQUIRE_FALSE(j["objects"][0].contains("embedding"));
}
