// face_landmark 逐脸复合用例：SCRFD 检测 → 逐脸裁剪 → 2d106det → 关键点回映原图。
//
// 背景：2d106det 的既定输入是「人脸裁剪图」，此前 Agent 对整帧直接估计 106 点，
// 只有人脸占满输入时语义才正确。本文件守护逐脸复合口径：
// 1) ConfigAdapter：复合解析（det_url→path 检测器、url|path→rec_path 关键点模型），
//    以及复用同任务 face_detection 条目（云端 FACE_LANDMARK pipeline 为两条独立模型）。
// 2) InferenceEngine：多脸图「每脸一组 106 点」，点数正确、有限、回映到各自人脸框内、逐脸不同。
// 3) 小脸回归：大帧中的小脸仍给出合理关键点。
// 4) 事件层：逐脸关键点归一化到 [0,1] 且逐脸不同。
// 5) 向后兼容：仅给关键点模型（无检测器）时退回整帧口径（点数仍为 106）。
#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <algorithm>
#include <array>
#include <cmath>
#include <fstream>
#include <limits>
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
// SCRFD 人脸检测器（face_detection）与 InsightFace 2d106 关键点模型
const char* kScrfd = "test_data/test_models/onnx/seetaface/scrfd_2.5g_bnkps_shape640x640.onnx";
const char* kLmkModel = "test_data/test_models/onnx/insightface/buffalo_l/2d106det.onnx";
// 单脸紧裁图（历史整帧 smoke 素材）与多人小脸大帧（3950×2706，SCRFD 可检出多张脸）
const char* kSingleFaceImage = "test_data/test_images/test_face_gender.jpg";
const char* kMultiFaceImage = "test_data/test_images/test_face_detection4.jpg";
constexpr size_t kLmkKeypointCount = 106;

bool file_nonempty(const char* path) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    return f.good() && f.tellg() > 0;
}

// 复合口径：path=SCRFD 检测器、rec_path=2d106det 关键点模型
ModelConfig make_composite_landmark_model() {
    ModelConfig m;
    m.name = "face_landmark";
    m.type = "face_landmark";
    m.backend = "ort";
    m.device = "cpu";
    m.path = kScrfd;
    m.rec_path = kLmkModel;
    m.input_size = {640, 640};
    m.confidence_threshold = 0.5f;
    return m;
}

// 旧口径：仅关键点模型（无检测器）→ 整帧估计（向后兼容路径）
ModelConfig make_legacy_landmark_model() {
    ModelConfig m;
    m.name = "face_landmark";
    m.type = "face_landmark";
    m.backend = "ort";
    m.device = "cpu";
    m.path = kLmkModel;
    m.input_size = {192, 192};
    return m;
}

// 关键点包围盒（像素坐标）
struct KpBounds {
    float x0 = 0.f, y0 = 0.f, x1 = 0.f, y1 = 0.f;
    float span_x() const { return x1 - x0; }
    float span_y() const { return y1 - y0; }
    float center_x() const { return (x0 + x1) * 0.5f; }
    float center_y() const { return (y0 + y1) * 0.5f; }
};

KpBounds keypoints_bounds(const std::vector<std::array<float, 3>>& kps) {
    KpBounds b;
    b.x0 = b.y0 = std::numeric_limits<float>::max();
    b.x1 = b.y1 = std::numeric_limits<float>::lowest();
    for (const auto& k : kps) {
        b.x0 = std::min(b.x0, k[0]);
        b.y0 = std::min(b.y0, k[1]);
        b.x1 = std::max(b.x1, k[0]);
        b.y1 = std::max(b.y1, k[1]);
    }
    return b;
}

// 关键点包围盒与检测框的交面积
float kp_box_intersection(const KpBounds& kp, const DetectionBox& b) {
    const float ix = std::max(0.f, std::min(kp.x1, b.x + b.w) - std::max(kp.x0, b.x));
    const float iy = std::max(0.f, std::min(kp.y1, b.y + b.h) - std::max(kp.y0, b.y));
    return ix * iy;
}

// 关键点包围盒落在检测框内的比例（交面积 / 关键点包围盒面积）。
// 若坐标为裁剪局部系（未回映原图），远离原点的脸该比例会趋近 0，可判定映射正确性。
float kp_inside_box_ratio(const KpBounds& kp, const DetectionBox& b) {
    const float kp_area = std::max(0.f, kp.span_x()) * std::max(0.f, kp.span_y());
    if (kp_area <= 1e-6f) return 0.f;
    return kp_box_intersection(kp, b) / kp_area;
}

// 「检测框 vs 关键点包围盒」重叠率（交面积 / 检测框面积），用于真机 smoke 合理性核对。
float box_overlap_ratio(const KpBounds& kp, const DetectionBox& b) {
    const float box_area = b.w * b.h;
    if (box_area <= 1e-6f) return 0.f;
    return kp_box_intersection(kp, b) / box_area;
}

float centroid_distance(const std::vector<std::array<float, 3>>& a,
                        const std::vector<std::array<float, 3>>& b) {
    const KpBounds ba = keypoints_bounds(a);
    const KpBounds bb = keypoints_bounds(b);
    const float dx = ba.center_x() - bb.center_x();
    const float dy = ba.center_y() - bb.center_y();
    return std::sqrt(dx * dx + dy * dy);
}
}  // namespace

// ── ConfigAdapter：复合口径 + 同任务 face_detection 复用 ──
TEST_CASE("ConfigAdapter parses composite face_landmark (det + landmark)", "[agent][config]") {
    ConfigAdapter adapter;
    std::string err;
    ModelConfig m;
    // det_url→path（检测器）；url→rec_path（关键点模型）
    REQUIRE(adapter.parse_model(
        json{{"name", "lmk"}, {"type", "face_landmark"}, {"det_url", kScrfd},
             {"url", kLmkModel}},
        &m, &err));
    REQUIRE(m.type == "face_landmark");
    REQUIRE(m.path == kScrfd);
    REQUIRE(m.rec_path == kLmkModel);
}

TEST_CASE("ConfigAdapter face_landmark reuses sibling face_detection model", "[agent][config]") {
    // 云端 FACE_LANDMARK pipeline 为 [face_detection, face_landmark] 两条独立模型：
    // face_landmark 未给 det_url 时，复用同任务 face_detection 路径作为检测器
    json j = json::parse(R"({
      "task_id": 71,
      "camera": {"id": 1, "name": "c", "url": "test_data/test_video60.mp4"},
      "models": [
        {"name": "fd", "type": "face_detection", "backend": "ort", "device": "cpu",
         "url": "test_data/test_models/onnx/seetaface/scrfd_2.5g_bnkps_shape640x640.onnx"},
        {"name": "lmk", "type": "face_landmark", "backend": "ort", "device": "cpu",
         "url": "test_data/test_models/onnx/insightface/buffalo_l/2d106det.onnx"}
      ],
      "events": {"transport": "http", "http": {"url": "http://x/cb", "token": "t"}}
    })");
    ConfigAdapter adapter;
    AdaptedTask t;
    std::string err;
    REQUIRE(adapter.from_json(j, &t, &err));
    REQUIRE(t.sdk.models.size() == 2);
    REQUIRE(t.sdk.models[1].type == "face_landmark");
    REQUIRE(t.sdk.models[1].path == kScrfd);     // 复用 face_detection 检测器
    REQUIRE(t.sdk.models[1].rec_path == kLmkModel);
}

// 无检测器时保持旧口径：path = 关键点模型、rec_path 为空（整帧兜底）
TEST_CASE("ConfigAdapter face_landmark without detector keeps landmark in path", "[agent][config]") {
    ConfigAdapter adapter;
    std::string err;
    ModelConfig m;
    REQUIRE(adapter.parse_model(
        json{{"name", "lmk"}, {"type", "face_landmark"}, {"url", kLmkModel}}, &m, &err));
    REQUIRE(m.path == kLmkModel);
    REQUIRE(m.rec_path.empty());
}

// ── 真机推理：多脸图逐脸 106 点 ──
TEST_CASE("InferenceEngine face_landmark yields one keypoint set per detected face",
          "[agent][model]") {
    if (!file_nonempty(kScrfd) || !file_nonempty(kLmkModel) || !file_nonempty(kMultiFaceImage))
        SKIP("face_landmark composite model/image absent");
    InferenceEngine eng;
    REQUIRE(eng.load(make_composite_landmark_model()));
    auto img = ImageData::imread(kMultiFaceImage);
    REQUIRE(!img.empty());
    InferResult r;
    REQUIRE(eng.infer(img, &r));
    REQUIRE(r.type == "face_landmark");
    REQUIRE(r.boxes.size() >= 2);   // 多人图：每脸一个对象

    std::vector<std::vector<std::array<float, 3>>> kp_sets;
    std::string per_face_report;
    size_t face_idx = 0;
    for (const auto& b : r.boxes) {
        REQUIRE(b.label_name == "face");
        REQUIRE(b.w > 0.f);
        REQUIRE(b.h > 0.f);
        REQUIRE(b.keypoints.size() == kLmkKeypointCount);
        for (const auto& k : b.keypoints) {
            REQUIRE(std::isfinite(k[0]));
            REQUIRE(std::isfinite(k[1]));
            REQUIRE(k[2] >= 0.f);
        }
        // 回映正确性：关键点包围盒应基本落在该脸的检测框内
        const KpBounds kb = keypoints_bounds(b.keypoints);
        REQUIRE(kb.span_x() > 0.f);
        REQUIRE(kb.span_y() > 0.f);
        const float ratio = kp_inside_box_ratio(kb, b);
        REQUIRE(ratio >= 0.6f);
        if (face_idx < 8)
            per_face_report += "#" + std::to_string(face_idx) + "(kp=" +
                               std::to_string(b.keypoints.size()) + ",inside=" +
                               std::to_string(ratio) + ",overlap=" +
                               std::to_string(box_overlap_ratio(kb, b)) + ") ";
        kp_sets.push_back(b.keypoints);
        ++face_idx;
    }
    WARN("face_landmark per-face smoke: faces=" << r.boxes.size()
         << " overlap=inter/det_box_area " << per_face_report);
    // 逐脸不同：任意两张脸的关键点中心不应重合（证明是逐脸裁剪而非复制同一组）
    float min_centroid_dist = std::numeric_limits<float>::max();
    for (size_t i = 0; i < kp_sets.size(); ++i)
        for (size_t j = i + 1; j < kp_sets.size(); ++j)
            min_centroid_dist = std::min(min_centroid_dist, centroid_distance(kp_sets[i], kp_sets[j]));
    REQUIRE(min_centroid_dist > 2.0f);
    WARN("face_landmark distinctness: min_pairwise_centroid_dist=" << min_centroid_dist);
}

// ── 小脸回归：大帧中的小脸仍给出合理关键点 ──
TEST_CASE("InferenceEngine face_landmark handles a small face in a large frame",
          "[agent][model]") {
    if (!file_nonempty(kScrfd) || !file_nonempty(kLmkModel) || !file_nonempty(kMultiFaceImage))
        SKIP("face_landmark composite model/image absent");
    InferenceEngine eng;
    REQUIRE(eng.load(make_composite_landmark_model()));
    auto img = ImageData::imread(kMultiFaceImage);
    REQUIRE(!img.empty());
    const float W = static_cast<float>(img.width());
    const float H = static_cast<float>(img.height());
    REQUIRE(W * H >= 1500000.f);   // 断言素材确为「大帧」

    InferResult r;
    REQUIRE(eng.infer(img, &r));
    REQUIRE_FALSE(r.boxes.empty());

    // 取面积最小的检出脸，验证小脸仍有点且点回映正确
    size_t smallest = 0;
    float smallest_area = std::numeric_limits<float>::max();
    for (size_t i = 0; i < r.boxes.size(); ++i) {
        const float area = r.boxes[i].w * r.boxes[i].h;
        if (area > 0.f && area < smallest_area) {
            smallest_area = area;
            smallest = i;
        }
    }
    const auto& b = r.boxes[smallest];
    const float ratio_of_frame = smallest_area / (W * H);
    REQUIRE(ratio_of_frame <= 0.02f);   // 明确是「小脸」（约占全帧面积 2% 以内）
    REQUIRE(b.keypoints.size() == kLmkKeypointCount);

    const KpBounds kb = keypoints_bounds(b.keypoints);
    REQUIRE(kb.span_x() > 0.f);
    REQUIRE(kb.span_y() > 0.f);
    const float inside = kp_inside_box_ratio(kb, b);
    REQUIRE(inside >= 0.6f);
    // 小脸关键点应落在该脸附近（而非帧原点/别处）
    REQUIRE(kb.center_x() >= -W * 0.1f);
    REQUIRE(kb.center_x() <= W * 1.1f);
    REQUIRE(kb.center_y() >= -H * 0.1f);
    REQUIRE(kb.center_y() <= H * 1.1f);
    WARN("face_landmark small-face smoke: faces=" << r.boxes.size()
                                                  << " smallest_box=(" << b.x << "," << b.y
                                                  << "," << b.w << "," << b.h
                                                  << ") area_ratio=" << ratio_of_frame
                                                  << " kp_inside_ratio=" << inside);
}

// ── 事件层：逐脸关键点归一化 [0,1] 且逐脸不同 ──
TEST_CASE("face_landmark per-face keypoints enter event normalized and distinct",
          "[agent][events]") {
    if (!file_nonempty(kScrfd) || !file_nonempty(kLmkModel) || !file_nonempty(kMultiFaceImage))
        SKIP("face_landmark composite model/image absent");
    InferenceEngine eng;
    REQUIRE(eng.load(make_composite_landmark_model()));
    auto img = ImageData::imread(kMultiFaceImage);
    REQUIRE(!img.empty());
    InferResult r;
    REQUIRE(eng.infer(img, &r));
    REQUIRE(r.boxes.size() >= 2);

    EventBus bus;
    std::vector<DetectionEvent> got;
    bus.set_sink([&](const DetectionEvent& e) { got.push_back(e); });
    EventMeta meta;
    meta.edge_code = "edge-01";
    meta.camera_id = 1;
    meta.task_id_num = 72;
    meta.alarm_interval_sec = 0;
    bus.register_task("t", meta);
    bus.on_detections("t", r.boxes, img.width(), img.height(), 1.0);

    REQUIRE(got.size() == 1);
    const auto j = got[0].to_json();
    REQUIRE(j["schema_version"] == 2);
    REQUIRE(j["objects"].size() == r.boxes.size());
    std::vector<std::array<float, 2>> centers;
    for (const auto& obj : j["objects"]) {
        REQUIRE(obj["keypoints"].size() == kLmkKeypointCount);
        float cx = 0.f, cy = 0.f;
        for (const auto& k : obj["keypoints"]) {
            REQUIRE(k.size() == 3);
            REQUIRE(k[0].get<float>() >= 0.f);
            REQUIRE(k[0].get<float>() <= 1.f);
            REQUIRE(k[1].get<float>() >= 0.f);
            REQUIRE(k[1].get<float>() <= 1.f);
            cx += k[0].get<float>();
            cy += k[1].get<float>();
        }
        centers.push_back({cx / static_cast<float>(kLmkKeypointCount),
                           cy / static_cast<float>(kLmkKeypointCount)});
    }
    // 逐脸不同：归一化关键点中心两两不同
    float min_dist = std::numeric_limits<float>::max();
    for (size_t i = 0; i < centers.size(); ++i)
        for (size_t j = i + 1; j < centers.size(); ++j) {
            const float dx = centers[i][0] - centers[j][0];
            const float dy = centers[i][1] - centers[j][1];
            min_dist = std::min(min_dist, std::sqrt(dx * dx + dy * dy));
        }
    REQUIRE(min_dist > 1e-3f);
    WARN("face_landmark event smoke: faces=" << centers.size() << " min_center_dist=" << min_dist);
}

// ── 向后兼容：无检测器 → 整帧口径（仍 106 点） ──
TEST_CASE("InferenceEngine face_landmark falls back to whole-frame without detector",
          "[agent][model]") {
    if (!file_nonempty(kLmkModel) || !file_nonempty(kSingleFaceImage))
        SKIP("face_landmark model/image absent");
    InferenceEngine eng;
    REQUIRE(eng.load(make_legacy_landmark_model()));
    auto img = ImageData::imread(kSingleFaceImage);
    REQUIRE(!img.empty());
    InferResult r;
    REQUIRE(eng.infer(img, &r));
    REQUIRE(r.type == "face_landmark");
    REQUIRE_FALSE(r.boxes.empty());
    for (const auto& b : r.boxes) {
        REQUIRE(b.keypoints.size() == kLmkKeypointCount);
        REQUIRE(b.w > 0.f);
        REQUIRE(b.h > 0.f);
    }
    WARN("face_landmark legacy fallback smoke: objects=" << r.boxes.size());
}
