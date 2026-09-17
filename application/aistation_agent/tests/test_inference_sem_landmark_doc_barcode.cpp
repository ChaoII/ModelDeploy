// sem / face_landmark / doc / barcode 四家族接线用例：
// 1) 类型名归一化（与云端场景目录 pipeline[].type 声明对齐：sem / face_landmark / doc / barcode）
// 2) 能力上报含四家族
// 3) ConfigAdapter 解析各家族模型条目（含 doc 的 OCR 多键、barcode 无权重）
// 4) 真机推理：
//    - sem：整帧对象 + attributes{类别名: 面积占比}（占比和 ≤ 1，含主导类）
//    - face_landmark：106 个关键点（归一化 0..1）
//    - doc：OCR 文本非空；barcode：ZXing 解码值非空
// 5) 事件契约 + JSON 往返
// 6) 真实 Pipeline → EventBus：sem 整帧占比对象、barcode 文本、face_landmark 关键点到达 sink
#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <set>
#include <string>
#include <thread>
#include <vector>

#include "capability.hpp"
#include "config_adapter.hpp"
#include "event_bus.hpp"
#include "inference_engine.hpp"
#include "nlohmann/json.hpp"
#include "pipeline.hpp"

using nlohmann::json;
using Catch::Approx;
using namespace modeldeploy::vision;

namespace {
const char* kSemModel = "test_data/test_models/onnx/yolo26n/yolo26n-sem.onnx";
const char* kSemImage = "test_data/test_images/bus.jpg";
const char* kLmkModel = "test_data/test_models/onnx/insightface/buffalo_l/2d106det.onnx";
const char* kLmkImage = "test_data/test_images/test_face_gender.jpg";
const char* kOcrDet = "test_data/test_models/onnx/ocr/ppocrv6_tiny/det_infer.onnx";
const char* kOcrCls = "test_data/test_models/onnx/ocr/ppocrv6_tiny/cls_infer.onnx";
const char* kOcrRec = "test_data/test_models/onnx/ocr/ppocrv6_tiny/rec_infer.onnx";
const char* kOcrDict = "test_data/ppocrv6_tiny_dict.txt";
const char* kDocImage = "test_data/test_images/ocr2.jpg";
const char* kQrImage = "test_data/qr_sample.png";
const char* kSemLoopVideo = "test_data/sem_bus_loop.mp4";
const char* kQrLoopVideo = "test_data/qr_code_loop.mp4";
// InsightFace 2d106det：106 点
constexpr size_t kLmkKeypointCount = 106;
// yolo26n-sem：cityscapes 19 类（模型头注释口径）
constexpr size_t kSemClasses = 19;

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

ModelConfig make_sem_model() {
    ModelConfig m;
    m.name = "sem";
    m.type = "sem";
    m.backend = "ort";
    m.device = "cpu";
    m.path = kSemModel;
    m.input_size = {640, 640};
    // cityscapes 19 类类别名（顺序与模型输出通道一致）
    m.labels = {"road", "sidewalk", "building", "wall", "fence", "pole", "traffic light",
                "traffic sign", "vegetation", "terrain", "sky", "person", "rider", "car",
                "truck", "bus", "train", "motorcycle", "bicycle"};
    return m;
}

ModelConfig make_landmark_model() {
    ModelConfig m;
    m.name = "face_landmark";
    m.type = "face_landmark";
    m.backend = "ort";
    m.device = "cpu";
    m.path = kLmkModel;
    m.input_size = {192, 192};
    return m;
}

ModelConfig make_doc_model() {
    ModelConfig m;
    m.name = "doc";
    m.type = "doc";
    m.backend = "ort";
    m.device = "cpu";
    m.path = kOcrDet;
    m.cls_path = kOcrCls;
    m.rec_path = kOcrRec;
    m.dict_path = kOcrDict;
    return m;
}

ModelConfig make_barcode_model() {
    ModelConfig m;
    m.name = "barcode";
    m.type = "barcode";
    m.backend = "ort";
    m.device = "cpu";
    // 原生 ZXing 解码：无需权重（path 可空）
    return m;
}

json barcode_task_model_json() {
    return json{{"name", "barcode"}, {"type", "barcode"}, {"backend", "ort"},
                {"device", "cpu"}, {"input_size", {640, 640}}};
}
}  // namespace

// ── 类型名归一化：与云端场景目录 pipeline[].type 对齐 + 常见别名 ──
TEST_CASE("normalize_model_type accepts sem aliases", "[agent][config]") {
    REQUIRE(normalize_model_type("sem") == "sem");
    REQUIRE(normalize_model_type("semantic_segmentation") == "sem");
    REQUIRE(normalize_model_type("semantic_seg") == "sem");
    REQUIRE(normalize_model_type("sem_seg") == "sem");
    REQUIRE(normalize_model_type("yolo_sem") == "sem");
    REQUIRE(normalize_model_type("sem_pipeline") == "sem");
}

TEST_CASE("normalize_model_type accepts face_landmark aliases", "[agent][config]") {
    REQUIRE(normalize_model_type("face_landmark") == "face_landmark");
    REQUIRE(normalize_model_type("face_landmarks") == "face_landmark");
    REQUIRE(normalize_model_type("face_lmk") == "face_landmark");
    REQUIRE(normalize_model_type("landmark") == "face_landmark");
    REQUIRE(normalize_model_type("face_keypoints") == "face_landmark");
    REQUIRE(normalize_model_type("2d106") == "face_landmark");
}

TEST_CASE("normalize_model_type accepts doc aliases", "[agent][config]") {
    REQUIRE(normalize_model_type("doc") == "doc");
    REQUIRE(normalize_model_type("doc_table") == "doc");
    REQUIRE(normalize_model_type("document") == "doc");
    REQUIRE(normalize_model_type("table_structure") == "doc");
    REQUIRE(normalize_model_type("ppstructure") == "doc");
}

TEST_CASE("normalize_model_type accepts barcode aliases", "[agent][config]") {
    REQUIRE(normalize_model_type("barcode") == "barcode");
    REQUIRE(normalize_model_type("barcodes") == "barcode");
    REQUIRE(normalize_model_type("qr") == "barcode");
    REQUIRE(normalize_model_type("qr_code") == "barcode");
    REQUIRE(normalize_model_type("bar_code") == "barcode");
}

TEST_CASE("detect_capabilities advertises sem/face_landmark/doc/barcode", "[agent][capability]") {
    auto c = detect_capabilities(8);
    std::set<std::string> fams;
    for (const auto& f : c["model_families"]) fams.insert(f.get<std::string>());
    REQUIRE(fams.count("sem") == 1);
    REQUIRE(fams.count("face_landmark") == 1);
    REQUIRE(fams.count("doc") == 1);
    REQUIRE(fams.count("barcode") == 1);
}

// ── ConfigAdapter：各家族条目解析（doc 走 OCR 多键；barcode 无权重）──
TEST_CASE("ConfigAdapter parses sem/face_landmark/doc/barcode models", "[agent][config]") {
    ConfigAdapter adapter;
    std::string err;

    ModelConfig sem;
    REQUIRE(adapter.parse_model(json{{"name", "sem"}, {"type", "sem"}, {"url", kSemModel}},
                                &sem, &err));
    REQUIRE(sem.type == "sem");
    REQUIRE(sem.path == kSemModel);

    ModelConfig lmk;
    REQUIRE(adapter.parse_model(
        json{{"name", "lmk"}, {"type", "face_landmark"}, {"url", kLmkModel}}, &lmk, &err));
    REQUIRE(lmk.type == "face_landmark");
    REQUIRE(lmk.path == kLmkModel);

    ModelConfig doc;
    REQUIRE(adapter.parse_model(json{{"name", "doc"}, {"type", "doc"}, {"det_url", kOcrDet},
                                     {"cls_url", kOcrCls}, {"rec_url", kOcrRec},
                                     {"dict_url", kOcrDict}},
                                &doc, &err));
    REQUIRE(doc.type == "doc");
    REQUIRE(doc.path == kOcrDet);
    REQUIRE(doc.cls_path == kOcrCls);
    REQUIRE(doc.rec_path == kOcrRec);
    REQUIRE(doc.dict_path == kOcrDict);

    // barcode：无 url/path 也应解析成功（原生解码，无权重）
    ModelConfig bc;
    REQUIRE(adapter.parse_model(barcode_task_model_json(), &bc, &err));
    REQUIRE(bc.type == "barcode");
    REQUIRE(bc.path.empty());

    // barcode：即便云端按目录 [ocr] 下发权重 url，也应接受（加载时忽略）
    ModelConfig bc2;
    REQUIRE(adapter.parse_model(
        json{{"name", "barcode"}, {"type", "barcode"}, {"url", kOcrDet}}, &bc2, &err));
    REQUIRE(bc2.type == "barcode");

    // 其它家族仍强制要求权重路径（不因 barcode 的放宽而回归）
    ModelConfig det;
    REQUIRE_FALSE(adapter.parse_model(json{{"name", "d"}, {"type", "detection"}}, &det, &err));
    REQUIRE_FALSE(err.empty());
}

// ── sem：真机推理 = 整帧对象 + attributes{类别名: 面积占比} ──
TEST_CASE("InferenceEngine sem yields full-frame object with class ratios", "[agent][model]") {
    if (!file_nonempty(kSemModel) || !file_nonempty(kSemImage)) SKIP("sem model/image absent");
    InferenceEngine eng;
    REQUIRE(eng.load(make_sem_model()));
    auto img = ImageData::imread(kSemImage);
    REQUIRE(!img.empty());
    InferResult r;
    REQUIRE(eng.infer(img, &r));
    REQUIRE(r.type == "sem");
    REQUIRE(r.boxes.size() == 1);

    const auto& b = r.boxes[0];
    // 整帧框（像素）→ 事件归一化后 0,0,1,1
    REQUIRE(b.x == 0.f);
    REQUIRE(b.y == 0.f);
    REQUIRE(b.w == static_cast<float>(img.width()));
    REQUIRE(b.h == static_cast<float>(img.height()));
    REQUIRE(b.label_id >= 0);
    REQUIRE(b.label_id < static_cast<int>(kSemClasses));
    REQUIRE_FALSE(b.label_name.empty());
    REQUIRE_FALSE(b.attributes.empty());

    float sum = 0.f, max_ratio = 0.f;
    std::string top_name, ratio_report;
    for (const auto& [k, v] : b.attributes) {
        REQUIRE(v >= 0.f);
        REQUIRE(v <= 1.f);
        sum += v;
        if (v > max_ratio) { max_ratio = v; top_name = k; }
        ratio_report += k + "=" + std::to_string(v) + " ";
    }
    // 每像素恰好归一类 → 占比和 == 1（浮点容差）；且必然有主导类
    REQUIRE(sum <= 1.0001f);
    REQUIRE(sum > 0.5f);
    // score = 主导类占比，且确实等于最大值
    REQUIRE(b.score == Approx(max_ratio));
    REQUIRE(top_name == b.label_name);
    WARN("sem smoke: classes=" << b.attributes.size() << " dominant=" << top_name
                               << " ratio=" << b.score << " sum=" << sum);
    WARN("sem ratios: " << ratio_report);
}

// sem 事件契约（不经模型）：整帧归一化 0,0,1,1 + 占比 attributes，且 JSON 往返保真
TEST_CASE("sem full-frame ratios reach event objects and round-trip", "[agent][events]") {
    EventBus bus;
    std::vector<DetectionEvent> got;
    bus.set_sink([&](const DetectionEvent& e) { got.push_back(e); });
    EventMeta meta;
    meta.edge_code = "edge-01";
    meta.camera_id = 1;
    meta.task_id_num = 101;
    meta.alarm_interval_sec = 0;
    bus.register_task("t", meta);

    DetectionBox b;
    b.x = 0.f; b.y = 0.f; b.w = 640.f; b.h = 480.f;
    b.score = 0.62f; b.label_id = 5; b.label_name = "bus";
    b.attributes = {{"bus", 0.62f}, {"road", 0.30f}, {"sky", 0.08f}};
    bus.on_detections("t", {b}, 640, 480, 3.0);

    REQUIRE(got.size() == 1);
    auto j = got[0].to_json();
    REQUIRE(j["objects"].size() == 1);
    const auto& obj = j["objects"][0];
    REQUIRE(obj["bbox"]["x"].get<float>() == Approx(0.f));
    REQUIRE(obj["bbox"]["y"].get<float>() == Approx(0.f));
    REQUIRE(obj["bbox"]["width"].get<float>() == Approx(1.f));
    REQUIRE(obj["bbox"]["height"].get<float>() == Approx(1.f));
    REQUIRE(obj["label"] == "bus");
    float sum = 0.f;
    for (auto it = obj["attributes"].begin(); it != obj["attributes"].end(); ++it) sum += it.value().get<float>();
    REQUIRE(sum <= Approx(1.0001f));
    REQUIRE(obj["attributes"]["bus"].get<float>() == Approx(0.62f));

    auto e2 = DetectionEvent::from_json(j);
    REQUIRE(e2.detections.size() == 1);
    REQUIRE(e2.detections[0].attributes.size() == 3);
    REQUIRE(e2.detections[0].attributes["road"] == Approx(0.30f));
}

// ── face_landmark：真机 106 点 ──
TEST_CASE("InferenceEngine face_landmark yields 106 keypoints", "[agent][model]") {
    if (!file_nonempty(kLmkModel) || !file_nonempty(kLmkImage))
        SKIP("face_landmark model/image absent");
    InferenceEngine eng;
    REQUIRE(eng.load(make_landmark_model()));
    auto img = ImageData::imread(kLmkImage);
    REQUIRE(!img.empty());
    InferResult r;
    REQUIRE(eng.infer(img, &r));
    REQUIRE(r.type == "face_landmark");
    REQUIRE_FALSE(r.boxes.empty());

    size_t total = 0;
    for (const auto& b : r.boxes) {
        REQUIRE_FALSE(b.label_name.empty());
        REQUIRE(b.keypoints.size() == kLmkKeypointCount);
        for (const auto& k : b.keypoints) {
            // 像素坐标（帧内，允许仿射反变换的轻微越界）
            REQUIRE(k[0] >= -static_cast<float>(img.width()) * 0.25f);
            REQUIRE(k[1] >= -static_cast<float>(img.height()) * 0.25f);
            REQUIRE(k[0] <= static_cast<float>(img.width()) * 1.25f);
            REQUIRE(k[1] <= static_cast<float>(img.height()) * 1.25f);
            REQUIRE(k[2] >= 0.f);
        }
        ++total;
    }
    REQUIRE(total >= 1);
    WARN("face_landmark smoke: objects=" << total << " keypoints/object="
                                         << kLmkKeypointCount);
}

// face_landmark 事件契约：关键点归一化到 0..1 并 JSON 往返
TEST_CASE("face_landmark keypoints normalized into event and round-trip", "[agent][events]") {
    EventBus bus;
    std::vector<DetectionEvent> got;
    bus.set_sink([&](const DetectionEvent& e) { got.push_back(e); });
    EventMeta meta;
    meta.task_id_num = 102;
    meta.alarm_interval_sec = 0;
    bus.register_task("t", meta);

    DetectionBox b;
    b.x = 0.f; b.y = 0.f; b.w = 248.f; b.h = 248.f;
    b.score = 1.0f; b.label_id = 0; b.label_name = "face";
    for (int i = 0; i < static_cast<int>(kLmkKeypointCount); ++i)
        b.keypoints.push_back({static_cast<float>(i) + 1.f, static_cast<float>(i) * 2.f, 0.f});
    bus.on_detections("t", {b}, 248, 248, 1.0);

    REQUIRE(got.size() == 1);
    auto j = got[0].to_json();
    REQUIRE(j["objects"].size() == 1);
    const auto& obj = j["objects"][0];
    REQUIRE(obj["keypoints"].size() == kLmkKeypointCount);
    for (const auto& k : obj["keypoints"]) {
        REQUIRE(k.size() == 3);
        REQUIRE(k[0].get<float>() >= 0.f);
        REQUIRE(k[0].get<float>() <= 1.f);
        REQUIRE(k[1].get<float>() >= 0.f);
        REQUIRE(k[1].get<float>() <= 1.f);
    }
    auto e2 = DetectionEvent::from_json(j);
    REQUIRE(e2.detections.size() == 1);
    REQUIRE(e2.detections[0].keypoints.size() == kLmkKeypointCount);
    REQUIRE(e2.detections[0].keypoints[105][0] == Approx(106.f / 248.f));
}

// ── doc：OCR 文本导出 ──
TEST_CASE("InferenceEngine doc extracts text via OCR pipeline", "[agent][model]") {
    if (!file_nonempty(kOcrDet) || !file_nonempty(kOcrCls) || !file_nonempty(kOcrRec) ||
        !file_nonempty(kOcrDict) || !file_nonempty(kDocImage))
        SKIP("doc OCR test data absent");
    InferenceEngine eng;
    REQUIRE(eng.load(make_doc_model()));
    auto img = ImageData::imread(kDocImage);
    REQUIRE(!img.empty());
    InferResult r;
    REQUIRE(eng.infer(img, &r));
    REQUIRE(r.type == "doc");
    REQUIRE_FALSE(r.boxes.empty());
    size_t with_text = 0;
    std::string sample;
    for (const auto& b : r.boxes) {
        if (b.text.empty()) continue;
        ++with_text;
        if (sample.empty()) sample = b.text;
        // 文本行包围盒在帧内
        REQUIRE(b.w > 0.f);
        REQUIRE(b.h > 0.f);
        REQUIRE(b.x >= 0.f);
        REQUIRE(b.y >= 0.f);
        REQUIRE(b.text_score > 0.f);
    }
    REQUIRE(with_text >= 1);
    WARN("doc smoke: text_lines=" << with_text << " sample=\"" << sample << "\"");
}

// ── barcode：原生解码 ──
TEST_CASE("InferenceEngine barcode decodes code value without weights", "[agent][model]") {
    if (!file_nonempty(kQrImage)) SKIP("barcode test image absent");
    InferenceEngine eng;
    REQUIRE(eng.load(make_barcode_model()));   // path 为空也须加载成功
    auto img = ImageData::imread(kQrImage);
    REQUIRE(!img.empty());
    InferResult r;
    REQUIRE(eng.infer(img, &r));
    REQUIRE(r.type == "barcode");
    REQUIRE_FALSE(r.boxes.empty());
    const auto& b = r.boxes[0];
    REQUIRE_FALSE(b.text.empty());          // 契约：objects[].text = 解码值
    REQUIRE_FALSE(b.label_name.empty());    // 码制名（QR_CODE 等）非空
    REQUIRE(b.x >= 0.f);
    REQUIRE(b.y >= 0.f);
    REQUIRE(b.w > 0.f);
    REQUIRE(b.h > 0.f);
    REQUIRE(b.x + b.w <= static_cast<float>(img.width()) + 1.f);
    REQUIRE(b.y + b.h <= static_cast<float>(img.height()) + 1.f);
    WARN("barcode smoke: format=" << b.label_name << " text=\"" << b.text
                                  << "\" codes=" << r.boxes.size());

    // 非码图：不得崩溃（可能 0 个码，不伪造对象）
    if (file_nonempty(kLmkImage)) {
        auto gray = ImageData::imread(kLmkImage);
        InferResult r2;
        REQUIRE(eng.infer(gray, &r2));
        WARN("barcode negative smoke: codes=" << r2.boxes.size());
    }
}

// doc/barcode 文本进入事件并 JSON 往返
TEST_CASE("doc/barcode text reaches event objects and round-trip", "[agent][events]") {
    EventBus bus;
    std::vector<DetectionEvent> got;
    bus.set_sink([&](const DetectionEvent& e) { got.push_back(e); });
    EventMeta meta;
    meta.task_id_num = 103;
    meta.alarm_interval_sec = 0;
    bus.register_task("t", meta);

    DetectionBox text_box;
    text_box.x = 10.f; text_box.y = 20.f; text_box.w = 100.f; text_box.h = 30.f;
    text_box.score = 0.9f; text_box.text = "HELLO-2026";
    text_box.text_score = 0.9f; text_box.label_name = "QR_CODE";
    bus.on_detections("t", {text_box}, 320, 240, 1.0);

    REQUIRE(got.size() == 1);
    auto j = got[0].to_json();
    REQUIRE(j["objects"][0]["text"] == "HELLO-2026");
    REQUIRE(j["objects"][0]["text_score"].get<float>() == Approx(0.9f));
    auto e2 = DetectionEvent::from_json(j);
    REQUIRE(e2.detections.size() == 1);
    REQUIRE(e2.detections[0].text == "HELLO-2026");
    REQUIRE(e2.detections[0].text_score == Approx(0.9f));
}

// ── 集成：barcode + face_landmark 经真实 Pipeline 到达事件 sink ──
TEST_CASE("Pipeline barcode+face_landmark deliver text and keypoints to sink",
          "[agent][pipeline][integration]") {
    if (!ensure_loop_video(kQrImage, kQrLoopVideo)) SKIP("barcode image/ffmpeg absent");
    if (!file_nonempty(kLmkModel)) SKIP("face_landmark model absent");

    TaskConfig cfg;
    cfg.id = "barcode_lmk_sink_test";
    cfg.input_url = kQrLoopVideo;
    cfg.enable_preview = false;
    cfg.models.push_back(make_barcode_model());
    cfg.models.push_back(make_landmark_model());

    EventBus bus;
    std::vector<DetectionEvent> got;
    bus.set_sink([&](const DetectionEvent& e) { got.push_back(e); });
    EventMeta meta;
    meta.edge_code = "edge-01";
    meta.camera_id = 1;
    meta.task_id_num = 104;
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
    bool found_text = false, found_kps = false;
    for (const auto& e : got) {
        const auto j = e.to_json();
        for (const auto& obj : j["objects"]) {
            if (obj.contains("text") && !obj["text"].get<std::string>().empty()) found_text = true;
            if (obj.contains("keypoints")) {
                REQUIRE(obj["keypoints"].size() == kLmkKeypointCount);
                for (const auto& k : obj["keypoints"]) {
                    REQUIRE(k[0].get<float>() >= 0.f);
                    REQUIRE(k[0].get<float>() <= 1.f);
                    REQUIRE(k[1].get<float>() >= 0.f);
                    REQUIRE(k[1].get<float>() <= 1.f);
                }
                found_kps = true;
            }
        }
    }
    REQUIRE(found_text);
    REQUIRE(found_kps);
    WARN("barcode+landmark pipeline smoke: events=" << got.size());
}

// ── 集成：sem 整帧占比对象经真实 Pipeline 到达事件 sink（验证 has_boxes 门控）──
TEST_CASE("Pipeline sem delivers full-frame ratio object to sink",
          "[agent][pipeline][integration]") {
    if (!file_nonempty(kSemModel)) SKIP("sem model absent");
    if (!ensure_loop_video(kSemImage, kSemLoopVideo)) SKIP("sem image/ffmpeg absent");

    TaskConfig cfg;
    cfg.id = "sem_sink_test";
    cfg.input_url = kSemLoopVideo;
    cfg.enable_preview = false;
    cfg.models.push_back(make_sem_model());

    EventBus bus;
    std::vector<DetectionEvent> got;
    bus.set_sink([&](const DetectionEvent& e) { got.push_back(e); });
    EventMeta meta;
    meta.edge_code = "edge-01";
    meta.camera_id = 1;
    meta.task_id_num = 105;
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
    const auto j = got[0].to_json();
    REQUIRE(j["objects"].size() >= 1);
    const auto& obj = j["objects"][0];
    REQUIRE(obj["bbox"]["x"].get<float>() == Approx(0.f));
    REQUIRE(obj["bbox"]["y"].get<float>() == Approx(0.f));
    REQUIRE(obj["bbox"]["width"].get<float>() == Approx(1.f));
    REQUIRE(obj["bbox"]["height"].get<float>() == Approx(1.f));
    REQUIRE(obj.contains("attributes"));
    float sum = 0.f;
    for (auto it = obj["attributes"].begin(); it != obj["attributes"].end(); ++it)
        sum += it.value().get<float>();
    REQUIRE(sum <= Approx(1.0001f));
    REQUIRE(sum > 0.5f);
    WARN("sem pipeline smoke: events=" << got.size() << " classes=" << obj["attributes"].size()
                                       << " sum=" << sum);
}
