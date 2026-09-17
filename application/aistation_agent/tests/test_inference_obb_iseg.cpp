#include <catch2/catch_test_macros.hpp>
#include <atomic>
#include <chrono>
#include <cstdlib>
#include <fstream>
#include <set>
#include <string>
#include <thread>

#include "capability.hpp"
#include "config_adapter.hpp"
#include "infer_group.hpp"
#include "inference_engine.hpp"
#include "nlohmann/json.hpp"
#include "pipeline.hpp"

using nlohmann::json;
using namespace modeldeploy::vision;

namespace {
const char* kObbModel = "test_data/test_models/onnx/yolo11n/yolo11n-obb_nms.onnx";
const char* kObbImage = "test_data/test_images/test_obb.jpg";
const char* kObbLoopVideo = "test_data/obb_dota_loop.mp4";
const char* kSegModel = "test_data/test_models/onnx/yolo11n/yolo11n-seg_nms.onnx";
const char* kSegImage = "test_data/test_images/test_person.jpg";
const char* kSegLoopVideo = "test_data/iseg_person_loop.mp4";

// 存在且非空才算可用（ffmpeg 失败会留下 0 字节文件，不能仅判 exists）。
bool file_nonempty(const char* path) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    return f.good() && f.tellg() > 0;
}

// 把静态图循环编码为短 mp4，供集成用例驱动真实解码 + 推理路径（沿用 LPR 用例的约定）。
// scale 过滤保证宽高为偶数（libx264 要求，测试图可能为奇数尺寸）。
bool ensure_loop_video(const char* img, const char* out) {
    if (file_nonempty(out)) return true;
    if (!file_nonempty(img)) return false;
    const std::string cmd = std::string("ffmpeg -y -loglevel error -loop 1 -i \"") + img +
                            "\" -vf \"scale=trunc(iw/2)*2:trunc(ih/2)*2\" -t 12 -r 8 "
                            "-pix_fmt yuv420p \"" + out + "\"";
    if (std::system(cmd.c_str()) != 0) return false;
    return file_nonempty(out);
}

ModelConfig make_model(const std::string& name, const std::string& type, const char* path) {
    ModelConfig m;
    m.name = name;
    m.type = type;
    m.backend = "ort";
    m.device = "cpu";
    m.path = path;
    m.input_size = {640, 640};
    m.confidence_threshold = 0.25f;
    return m;
}
}  // namespace

// ── 类型名归一化：与云端场景目录声明（type="obb" / type="iseg"）对齐 ──
TEST_CASE("normalize_model_type accepts obb and iseg aliases", "[agent][config]") {
    REQUIRE(normalize_model_type("obb") == "obb");
    REQUIRE(normalize_model_type("obb_det") == "obb");
    REQUIRE(normalize_model_type("iseg") == "iseg");
    REQUIRE(normalize_model_type("seg") == "iseg");                    // 云端编排关键词映射
    REQUIRE(normalize_model_type("instance_segmentation") == "iseg");
}

TEST_CASE("detect_capabilities advertises obb and iseg families", "[agent][capability]") {
    auto c = detect_capabilities(8);
    std::set<std::string> fams;
    for (const auto& f : c["model_families"]) fams.insert(f.get<std::string>());
    REQUIRE(fams.count("obb") == 1);
    REQUIRE(fams.count("iseg") == 1);
}

TEST_CASE("ConfigAdapter parses obb task model", "[agent][config]") {
    json j = json::parse(R"({
      "task_id": 21,
      "camera": {"id": 1, "name": "c", "url": "test_data/test_video60.mp4"},
      "models": [{"name": "obb", "type": "obb", "backend": "ort", "device": "cpu",
                  "url": "test_data/test_models/onnx/yolo11n/yolo11n-obb_nms.onnx",
                  "input_size": [640, 640], "confidence_threshold": 0.25}],
      "events": {"transport": "http", "http": {"url": "http://x/cb", "token": "t"}}
    })");
    ConfigAdapter adapter;
    AdaptedTask t;
    std::string err;
    REQUIRE(adapter.from_json(j, &t, &err));
    REQUIRE(t.sdk.models.size() == 1);
    REQUIRE(t.sdk.models[0].type == "obb");
    REQUIRE(t.sdk.models[0].path == kObbModel);
}

TEST_CASE("ConfigAdapter parses iseg task model via seg alias", "[agent][config]") {
    json j = json::parse(R"({
      "task_id": 22,
      "camera": {"id": 1, "name": "c", "url": "test_data/test_video60.mp4"},
      "models": [{"name": "iseg", "type": "iseg", "backend": "ort", "device": "cpu",
                  "url": "test_data/test_models/onnx/yolo11n/yolo11n-seg_nms.onnx",
                  "input_size": [640, 640], "confidence_threshold": 0.25}],
      "events": {"transport": "http", "http": {"url": "http://x/cb", "token": "t"}}
    })");
    ConfigAdapter adapter;
    AdaptedTask t;
    std::string err;
    REQUIRE(adapter.from_json(j, &t, &err));
    REQUIRE(t.sdk.models[0].type == "iseg");
    REQUIRE(t.sdk.models[0].path == kSegModel);
}

// ── 真机 smoke：加载模型 + 匹配测试图，断言非空结果（旋转框降维为外接 bbox）──
TEST_CASE("InferenceEngine obb predicts axis-aligned boxes on real image", "[agent][model]") {
    if (!std::ifstream(kObbModel).good() || !std::ifstream(kObbImage).good())
        SKIP("OBB model/image absent");
    InferenceEngine eng;
    REQUIRE(eng.load(make_model("obb", "obb", kObbModel)));
    auto img = ImageData::imread(kObbImage);
    REQUIRE(!img.empty());
    InferResult r;
    REQUIRE(eng.infer(img, &r));
    REQUIRE(r.type == "obb");
    REQUIRE_FALSE(r.boxes.empty());
    for (const auto& b : r.boxes) {
        REQUIRE(b.w > 0.f);
        REQUIRE(b.h > 0.f);
        REQUIRE(b.score > 0.f);
        REQUIRE_FALSE(b.label_name.empty());          // 事件 label 必须非空（object_present 依赖）
        // 外接框应大致落在帧内（边缘目标旋转框可略微越界，与 detection 路径一致不做裁剪）
        REQUIRE(b.x >= -static_cast<float>(img.width()) * 0.2f);
        REQUIRE(b.y >= -static_cast<float>(img.height()) * 0.2f);
        REQUIRE(b.x + b.w <= static_cast<float>(img.width()) * 1.2f);
        REQUIRE(b.y + b.h <= static_cast<float>(img.height()) * 1.2f);
    }
}

TEST_CASE("InferenceEngine iseg predicts instance boxes on real image", "[agent][model]") {
    if (!std::ifstream(kSegModel).good() || !std::ifstream(kSegImage).good())
        SKIP("ISeg model/image absent");
    InferenceEngine eng;
    REQUIRE(eng.load(make_model("iseg", "iseg", kSegModel)));
    auto img = ImageData::imread(kSegImage);
    REQUIRE(!img.empty());
    InferResult r;
    REQUIRE(eng.infer(img, &r));
    REQUIRE(r.type == "iseg");
    REQUIRE_FALSE(r.boxes.empty());
    for (const auto& b : r.boxes) {
        REQUIRE(b.w > 0.f);
        REQUIRE(b.h > 0.f);
        REQUIRE(b.score > 0.f);
        REQUIRE_FALSE(b.label_name.empty());
    }
}

// obb/iseg 无 det_model()，必须走 non_det 通道（detection 通道不受影响）
TEST_CASE("InferGroup routes obb/iseg through non-det path", "[agent][model]") {
    if (!std::ifstream(kObbModel).good() || !std::ifstream(kSegModel).good() ||
        !std::ifstream(kObbImage).good())
        SKIP("OBB/ISeg model absent");
    InferGroup g;
    REQUIRE(g.load_models({make_model("obb", "obb", kObbModel),
                           make_model("iseg", "iseg", kSegModel)}, nullptr));
    auto img = ImageData::imread(kObbImage);
    REQUIRE(!img.empty());
    std::vector<std::pair<std::string, std::vector<DetectionResult>>> sdk_dets;
    std::vector<std::pair<std::string, InferResult>> non_det;
    g.run_models(img, &sdk_dets, &non_det);
    REQUIRE(sdk_dets.empty());
    REQUIRE(non_det.size() == 2);
}

// ── 集成：真实 Pipeline 把 obb/iseg 框送入 detection sink（云端 object_present 数据源）──
TEST_CASE("Pipeline obb delivers boxes to sink", "[agent][pipeline][integration]") {
    if (!ensure_loop_video(kObbImage, kObbLoopVideo)) SKIP("OBB image/ffmpeg absent");
    TaskConfig cfg;
    cfg.id = "obb_sink_test";
    cfg.input_url = kObbLoopVideo;
    cfg.enable_preview = false;
    cfg.models.push_back(make_model("obb", "obb", kObbModel));

    Pipeline pipe(cfg);
    std::atomic<int> boxes{0};
    pipe.set_detection_sink([&](const std::vector<DetectionBox>& b, int w, int h, double) {
        REQUIRE(w > 0);
        REQUIRE(h > 0);
        for (const auto& x : b)
            if (!x.label_name.empty()) ++boxes;
    });
    REQUIRE(pipe.start());
    for (int i = 0; i < 400 && boxes.load() == 0; ++i)
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
    pipe.stop();
    REQUIRE(boxes.load() > 0);
}

TEST_CASE("Pipeline iseg delivers boxes to sink", "[agent][pipeline][integration]") {
    if (!ensure_loop_video(kSegImage, kSegLoopVideo)) SKIP("ISeg image/ffmpeg absent");
    TaskConfig cfg;
    cfg.id = "iseg_sink_test";
    cfg.input_url = kSegLoopVideo;
    cfg.enable_preview = false;
    cfg.models.push_back(make_model("iseg", "iseg", kSegModel));

    Pipeline pipe(cfg);
    std::atomic<int> boxes{0};
    pipe.set_detection_sink([&](const std::vector<DetectionBox>& b, int w, int h, double) {
        REQUIRE(w > 0);
        REQUIRE(h > 0);
        for (const auto& x : b)
            if (!x.label_name.empty()) ++boxes;
    });
    REQUIRE(pipe.start());
    for (int i = 0; i < 400 && boxes.load() == 0; ++i)
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
    pipe.stop();
    REQUIRE(boxes.load() > 0);
}
