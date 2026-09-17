// B2a 家族接线用例：face_attr（性别/年龄属性）、face_as（活体）、depth（深度）
// 1) 类型名归一化（云端场景目录声明 type="face_attr"/"face_as"/"depth" + 别名）
// 2) 能力上报含三个新族
// 3) ConfigAdapter 复合模型路径解析（det/cls/rec 子模型）
// 4) 真机推理：face_attr → 人脸框 attributes{gender_male, age_young}（分数 0..1）
//    face_as → attributes{liveness}（0..1）；depth → 整帧对象 depth>0（米）
// 5) 事件契约：attributes/depth 写入 objects[]，JSON 往返保真；无深度时不写 depth 字段
#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <map>
#include <set>
#include <string>

#include "capability.hpp"
#include "config_adapter.hpp"
#include "event_bus.hpp"
#include "inference_engine.hpp"
#include "nlohmann/json.hpp"

using nlohmann::json;
using Catch::Approx;
using namespace modeldeploy::vision;

namespace {
const char* kFaceImg = "test_data/test_images/test_face_gender.jpg";
const char* kFaceVideo = "test_data/test_video60.mp4";         // 含可检测人脸的稳定素材
const char* kFaceFrame = "test_data/face_attr_frame.jpg";      // 由视频抽帧生成
const char* kScrfd = "test_data/test_models/onnx/seetaface/scrfd_2.5g_bnkps_shape640x640.onnx";
const char* kGender = "test_data/test_models/onnx/seetaface/gender_predictor.onnx";
const char* kAge = "test_data/test_models/onnx/seetaface/age_predictor.onnx";
const char* kFasFirst = "test_data/test_models/onnx/seetaface/fas_first.onnx";
const char* kFasSecond = "test_data/test_models/onnx/seetaface/fas_second.onnx";
const char* kDepthModel = "test_data/test_models/onnx/yolo26n/yolo26n-depth.onnx";
const char* kDepthImg = "test_data/test_images/test_depth_540.jpg";

bool exists(const char* p) { return std::ifstream(p).good(); }

// 从含人脸的视频抽取首帧作为静图（test_data 不入库，测试时按需生成）；
// 现成静图作为兜底（部分紧裁人脸图 SCRFD 不易检出）。
const char* ensure_face_image() {
    if (exists(kFaceFrame)) return kFaceFrame;
    if (!exists(kFaceVideo)) return kFaceImg;
    const std::string cmd = std::string("ffmpeg -y -loglevel error -ss 5 -i \"") + kFaceVideo +
                            "\" -frames:v 1 \"" + kFaceFrame + "\"";
    if (std::system(cmd.c_str()) != 0) return kFaceImg;
    return exists(kFaceFrame) ? kFaceFrame : kFaceImg;
}

ModelConfig make_face_attr_model() {
    ModelConfig m;
    m.name = "face_attr";
    m.type = "face_attr";
    m.backend = "ort";
    m.device = "cpu";
    m.path = kScrfd;        // det
    m.cls_path = kGender;   // 性别
    m.rec_path = kAge;      // 年龄
    m.input_size = {640, 640};
    m.confidence_threshold = 0.3f;
    return m;
}

ModelConfig make_face_as_model() {
    ModelConfig m;
    m.name = "face_as";
    m.type = "face_as";
    m.backend = "ort";
    m.device = "cpu";
    m.path = kScrfd;      // det
    m.cls_path = kFasFirst;
    m.rec_path = kFasSecond;
    m.input_size = {640, 640};
    m.confidence_threshold = 0.3f;
    return m;
}

ModelConfig make_depth_model() {
    ModelConfig m;
    m.name = "depth";
    m.type = "depth";
    m.backend = "ort";
    m.device = "cpu";
    m.path = kDepthModel;
    m.input_size = {640, 640};
    return m;
}
}  // namespace

// ── 类型名归一化 ──
TEST_CASE("normalize_model_type accepts face_attr/face_as/depth aliases", "[agent][config]") {
    REQUIRE(normalize_model_type("face_attr") == "face_attr");
    REQUIRE(normalize_model_type("face_attribute") == "face_attr");
    REQUIRE(normalize_model_type("face_attributes") == "face_attr");
    REQUIRE(normalize_model_type("face_as") == "face_as");
    REQUIRE(normalize_model_type("face_antispoof") == "face_as");
    REQUIRE(normalize_model_type("antispoof") == "face_as");
    REQUIRE(normalize_model_type("face_liveness") == "face_as");
    REQUIRE(normalize_model_type("liveness") == "face_as");
    REQUIRE(normalize_model_type("depth") == "depth");
    REQUIRE(normalize_model_type("depth_estimation") == "depth");
    REQUIRE(normalize_model_type("mono_depth") == "depth");
}

TEST_CASE("detect_capabilities advertises face_attr/face_as/depth families", "[agent][capability]") {
    auto c = detect_capabilities(8);
    std::set<std::string> fams;
    for (const auto& f : c["model_families"]) fams.insert(f.get<std::string>());
    REQUIRE(fams.count("face_attr") == 1);
    REQUIRE(fams.count("face_as") == 1);
    REQUIRE(fams.count("depth") == 1);
}

// ── ConfigAdapter：复合模型子路径解析（det/cls/rec） ──
TEST_CASE("ConfigAdapter parses face_attr/face_as/depth task models", "[agent][config]") {
    json j = json::parse(R"({
      "task_id": 41,
      "camera": {"id": 1, "name": "c", "url": "test_data/test_video60.mp4"},
      "models": [
        {"name": "fa", "type": "face_attr", "backend": "ort", "device": "cpu",
         "det_url": "test_data/test_models/onnx/seetaface/scrfd_2.5g_bnkps_shape640x640.onnx",
         "cls_url": "test_data/test_models/onnx/seetaface/gender_predictor.onnx",
         "rec_url": "test_data/test_models/onnx/seetaface/age_predictor.onnx",
         "attributes": ["gender_male", "age_young"]},
        {"name": "fas", "type": "face_as", "backend": "ort", "device": "cpu",
         "det_url": "test_data/test_models/onnx/seetaface/scrfd_2.5g_bnkps_shape640x640.onnx",
         "first_url": "test_data/test_models/onnx/seetaface/fas_first.onnx",
         "second_url": "test_data/test_models/onnx/seetaface/fas_second.onnx"},
        {"name": "d", "type": "depth", "backend": "ort", "device": "cpu",
         "url": "test_data/test_models/onnx/yolo26n/yolo26n-depth.onnx"}
      ],
      "events": {"transport": "http", "http": {"url": "http://x/cb", "token": "t"}}
    })");
    ConfigAdapter adapter;
    AdaptedTask t;
    std::string err;
    REQUIRE(adapter.from_json(j, &t, &err));
    REQUIRE(t.sdk.models.size() == 3);

    REQUIRE(t.sdk.models[0].type == "face_attr");
    REQUIRE(t.sdk.models[0].path == kScrfd);
    REQUIRE(t.sdk.models[0].cls_path == kGender);
    REQUIRE(t.sdk.models[0].rec_path == kAge);
    REQUIRE(t.sdk.models[0].attributes.size() == 2);

    REQUIRE(t.sdk.models[1].type == "face_as");
    REQUIRE(t.sdk.models[1].path == kScrfd);
    REQUIRE(t.sdk.models[1].cls_path == kFasFirst);
    REQUIRE(t.sdk.models[1].rec_path == kFasSecond);

    REQUIRE(t.sdk.models[2].type == "depth");
    REQUIRE(t.sdk.models[2].path == kDepthModel);
}

// ── 真机推理：face_attr（性别/年龄概率） ──
TEST_CASE("InferenceEngine face_attr yields gender/age attributes on real image",
          "[agent][model]") {
    if (!exists(kScrfd) || !exists(kGender) || !exists(kAge)) SKIP("face_attr model absent");
    const char* face_img = ensure_face_image();
    if (!exists(face_img)) SKIP("face_attr image absent");
    InferenceEngine eng;
    REQUIRE(eng.load(make_face_attr_model()));
    auto img = ImageData::imread(face_img);
    REQUIRE(!img.empty());
    InferResult r;
    REQUIRE(eng.infer(img, &r));
    REQUIRE(r.type == "face_attr");
    REQUIRE_FALSE(r.boxes.empty());

    size_t faces = 0;
    for (const auto& b : r.boxes) {
        REQUIRE(b.w > 0.f);
        REQUIRE(b.h > 0.f);
        REQUIRE_FALSE(b.label_name.empty());
        // 默认属性名与云端 FACE_ATTR 契约一致
        std::map<std::string, float> attrs;
        for (const auto& [k, v] : b.attributes) attrs[k] = v;
        REQUIRE(attrs.count("gender_male") == 1);
        REQUIRE(attrs.count("age_young") == 1);
        REQUIRE(attrs["gender_male"] >= 0.f);
        REQUIRE(attrs["gender_male"] <= 1.f);
        REQUIRE(attrs["age_young"] >= 0.f);
        REQUIRE(attrs["age_young"] <= 1.f);
        if (faces == 0)
            WARN("face_attr sample: gender_male=" << attrs["gender_male"]
                                                  << " age_young=" << attrs["age_young"]);
        ++faces;
    }
    WARN("face_attr smoke: faces=" << faces);
}

// ── 真机推理：face_as（活体分数） ──
TEST_CASE("InferenceEngine face_as yields liveness attribute on real image", "[agent][model]") {
    if (!exists(kScrfd) || !exists(kFasFirst)) SKIP("face_as model absent");
    const char* face_img = ensure_face_image();
    if (!exists(face_img)) SKIP("face_as image absent");
    InferenceEngine eng;
    REQUIRE(eng.load(make_face_as_model()));
    auto img = ImageData::imread(face_img);
    REQUIRE(!img.empty());
    InferResult r;
    REQUIRE(eng.infer(img, &r));
    REQUIRE(r.type == "face_as");
    REQUIRE_FALSE(r.boxes.empty());

    size_t faces = 0;
    for (const auto& b : r.boxes) {
        REQUIRE(b.w > 0.f);
        REQUIRE(b.h > 0.f);
        std::map<std::string, float> attrs;
        for (const auto& [k, v] : b.attributes) attrs[k] = v;
        REQUIRE(attrs.count("liveness") == 1);
        REQUIRE(attrs["liveness"] >= 0.f);
        REQUIRE(attrs["liveness"] <= 1.f);
        if (faces == 0) WARN("face_as sample: liveness=" << attrs["liveness"]);
        ++faces;
    }
    WARN("face_as smoke: faces=" << faces);
}

// ── 真机推理：depth（最近距离，米） ──
TEST_CASE("InferenceEngine depth yields metric depth on real image", "[agent][model]") {
    if (!exists(kDepthModel) || !exists(kDepthImg)) SKIP("depth model/image absent");
    InferenceEngine eng;
    REQUIRE(eng.load(make_depth_model()));
    auto img = ImageData::imread(kDepthImg);
    REQUIRE(!img.empty());
    InferResult r;
    REQUIRE(eng.infer(img, &r));
    REQUIRE(r.type == "depth");
    REQUIRE(r.boxes.size() == 1);
    const auto& b = r.boxes[0];
    REQUIRE(b.has_depth);
    REQUIRE(std::isfinite(b.depth));
    REQUIRE(b.depth > 0.f);
    WARN("depth smoke: near_m=" << b.depth);
}

// ── 事件契约：attributes/depth 写入 objects[]，JSON 往返保真 ──
TEST_CASE("face_attr/face_as attributes and depth enter event objects with JSON round-trip",
          "[agent][events]") {
    EventBus bus;
    std::vector<DetectionEvent> got;
    bus.set_sink([&](const DetectionEvent& e) { got.push_back(e); });
    EventMeta meta;
    meta.edge_code = "edge-01";
    meta.camera_id = 1;
    meta.task_id_num = 91;
    meta.alarm_interval_sec = 0;
    bus.register_task("t", meta);

    DetectionBox face;
    face.x = 100.f; face.y = 50.f; face.w = 200.f; face.h = 200.f;
    face.score = 0.9f; face.label_id = 0; face.label_name = "face";
    face.attributes = {{"gender_male", 0.91f}, {"age_young", 0.83f}};

    DetectionBox dist;
    dist.x = 0.f; dist.y = 0.f; dist.w = 640.f; dist.h = 480.f;
    dist.label_id = -1; dist.label_name = "depth";
    dist.has_depth = true; dist.depth = 2.35f;

    bus.on_detections("t", {face, dist}, 640, 480, 1.0);
    REQUIRE(got.size() == 1);

    auto j = got[0].to_json();
    REQUIRE(j["objects"].size() == 2);
    REQUIRE(j["objects"][0].contains("attributes"));
    REQUIRE(j["objects"][0]["attributes"]["gender_male"].get<float>() == Approx(0.91f));
    REQUIRE(j["objects"][0]["attributes"]["age_young"].get<float>() == Approx(0.83f));
    REQUIRE(j["objects"][1].contains("depth"));
    REQUIRE(j["objects"][1]["depth"].get<float>() == Approx(2.35f));
    REQUIRE(j["schema_version"] == 2);

    auto e2 = DetectionEvent::from_json(j);
    REQUIRE(e2.detections.size() == 2);
    REQUIRE(e2.detections[0].attributes.at("gender_male") == Approx(0.91f));
    REQUIRE(e2.detections[0].attributes.at("age_young") == Approx(0.83f));
    REQUIRE(e2.detections[1].has_depth);
    REQUIRE(e2.detections[1].depth == Approx(2.35f));
}

// 无深度/无属性的普通检测事件不新增 depth/attributes 字段（契约向后兼容）
TEST_CASE("event without depth omits depth field", "[agent][events]") {
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
    REQUIRE_FALSE(j["objects"][0].contains("depth"));
    REQUIRE_FALSE(j["objects"][0].contains("attributes"));
}
