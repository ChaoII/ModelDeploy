#include <catch2/catch_test_macros.hpp>
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdlib>
#include <fstream>
#include <map>
#include <mutex>
#include <set>
#include <thread>
#include "pipeline.hpp"

static const char* kVideo = "test_data/test_video60.mp4";
static const char* kModel = "test_data/test_models/onnx/yolo11n/yolo11n_nms.onnx";

// 生成 LPR 集成用例所需的回环视频：把 LPR 测试图循环编码成短 mp4（视频解码为 CPU NV12，
// 正是崩溃路径）。视频已存在则直接复用；ffmpeg 不在 PATH 时返回 false，用例 SKIP。
static bool ensure_lpr_video(const char* img, const char* out) {
    if (std::ifstream(out).good()) return true;
    if (!std::ifstream(img).good()) return false;
    const std::string cmd = std::string("ffmpeg -y -loglevel error -loop 1 -i \"") + img +
                            "\" -t 20 -r 8 -pix_fmt yuv420p \"" + out + "\"";
    if (std::system(cmd.c_str()) != 0) return false;
    return std::ifstream(out).good();
}

TEST_CASE("Pipeline detection sink fires on real detections", "[agent][pipeline][integration]") {
    if (!std::ifstream(kVideo).good() || !std::ifstream(kModel).good())
        SKIP("test video/model absent");

    TaskConfig cfg;
    cfg.id = "sink_test";
    cfg.input_url = kVideo;
    cfg.enable_preview = false;
    ModelConfig m;
    m.name = "det"; m.type = "detection"; m.backend = "ort"; m.device = "cpu";
    m.path = kModel; m.labels = {"person"}; m.input_size = {640, 640};
    cfg.models.push_back(m);

    Pipeline pipe(cfg);
    std::atomic<int> calls{0};
    std::atomic<int> boxes{0};
    pipe.set_detection_sink([&](const std::vector<DetectionBox>& b, int w, int h, double) {
        ++calls; boxes += static_cast<int>(b.size());
        REQUIRE(w > 0); REQUIRE(h > 0);
    });
    REQUIRE(pipe.start());
    for (int i = 0; i < 100 && calls.load() == 0; ++i)
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
    pipe.stop();
    REQUIRE(calls.load() > 0);
    REQUIRE(boxes.load() > 0);
}

// 跟踪集成用例：启用 tracking 时，Pipeline 逐帧跑 ByteTracker 并把 track_id 回填到
// 检测框。断言 sink 至少收到一个 track_id>=0 的框，且同一 track_id 跨多帧保持一致
// （同一目标稳定绑定同一 id）。test_data 缺失则 SKIP。
TEST_CASE("Pipeline tracking backfills stable track_id", "[agent][pipeline][integration]") {
    if (!std::ifstream(kVideo).good() || !std::ifstream(kModel).good())
        SKIP("test video/model absent");

    TaskConfig cfg;
    cfg.id = "track_test";
    cfg.input_url = kVideo;
    cfg.enable_preview = false;
    cfg.tracking.enabled = true;
    cfg.tracking.algorithm = "bytetrack";
    ModelConfig m;
    m.name = "det"; m.type = "detection"; m.backend = "ort"; m.device = "cpu";
    m.path = kModel; m.labels = {"person"}; m.input_size = {640, 640};
    cfg.models.push_back(m);

    Pipeline pipe(cfg);
    std::atomic<int> tracked_boxes{0};
    std::mutex mtx;
    std::map<int, int> id_frame_count;   // track_id -> 出现该 id 的帧数（每帧去重）
    pipe.set_detection_sink([&](const std::vector<DetectionBox>& boxes, int w, int h, double) {
        REQUIRE(w > 0);
        REQUIRE(h > 0);
        std::set<int> ids;
        for (const auto& b : boxes) {
            if (b.track_id >= 0) { ++tracked_boxes; ids.insert(b.track_id); }
        }
        std::lock_guard<std::mutex> lock(mtx);
        for (int id : ids) ++id_frame_count[id];
    });
    REQUIRE(pipe.start());
    // 等待累计若干个带 id 的框（覆盖多帧），上限 30s
    for (int i = 0; i < 600 && tracked_boxes.load() < 4; ++i)
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
    pipe.stop();

    REQUIRE(tracked_boxes.load() > 0);
    // 同一目标跨帧保持同一 id：至少一个 track_id 在多帧出现
    int max_frames = 0;
    {
        std::lock_guard<std::mutex> lock(mtx);
        for (const auto& [id, n] : id_frame_count) max_frames = std::max(max_frames, n);
    }
    REQUIRE(max_frames >= 2);
}

// 属性 pipeline 送达 sink 的集成用例：驱动真实 Pipeline，断言检测 sink 收到
// 携带非空 attributes 的 DetectionBox（覆盖 detect_loop 中 non_det → sink 的合并分支）。
TEST_CASE("Pipeline pedestrian_attribute delivers attributes to sink",
          "[agent][pipeline][integration]") {
    if (!std::ifstream(kVideo).good() ||
        !std::ifstream("test_data/test_models/onnx/zhgd_det.onnx").good() ||
        !std::ifstream("test_data/test_models/onnx/zhgd_ml.onnx").good())
        SKIP("test data absent");

    TaskConfig cfg;
    cfg.id = "attr_sink_test";
    cfg.input_url = kVideo;
    cfg.enable_preview = false;
    ModelConfig m;
    m.name = "pa";
    m.type = "pedestrian_attribute";
    m.backend = "ort";
    m.device = "cpu";
    m.path = "test_data/test_models/onnx/zhgd_det.onnx";
    m.rec_path = "test_data/test_models/onnx/zhgd_ml.onnx";
    m.input_size = {1280, 1280};
    m.confidence_threshold = 0.5f;
    m.attributes = {"safety_helmet", "reflective_vest", "safety_rope", "work_uniform"};
    cfg.models.push_back(m);

    Pipeline pipe(cfg);
    std::atomic<int> attr_boxes{0};
    pipe.set_detection_sink([&](const std::vector<DetectionBox>& boxes, int w, int h, double) {
        REQUIRE(w > 0);
        REQUIRE(h > 0);
        for (const auto& b : boxes)
            if (!b.attributes.empty()) ++attr_boxes;
    });
    REQUIRE(pipe.start());
    // 有界等待（模型加载 + 推理，上限 20s）
    for (int i = 0; i < 400 && attr_boxes.load() == 0; ++i)
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
    pipe.stop();
    REQUIRE(attr_boxes.load() > 0);
}

// OCR pipeline 送达 sink 的集成用例：驱动真实 Pipeline（视频/多帧路径），断言检测
// sink 收到带非空 text 的 DetectionBox。单图 infer 已由 test_inference_ocr 覆盖，
// 本用例专门覆盖视频帧（解码格式/重复推理）路径。
TEST_CASE("Pipeline ocr delivers text to sink",
          "[agent][pipeline][integration]") {
    if (!std::ifstream("test_data/ocr_text_loop.mp4").good() ||
        !std::ifstream("test_data/test_models/onnx/ocr/ppocrv6_tiny/det_infer.onnx").good() ||
        !std::ifstream("test_data/test_models/onnx/ocr/ppocrv6_tiny/cls_infer.onnx").good() ||
        !std::ifstream("test_data/test_models/onnx/ocr/ppocrv6_tiny/rec_infer.onnx").good() ||
        !std::ifstream("test_data/ppocrv6_tiny_dict.txt").good())
        SKIP("OCR test data absent");

    TaskConfig cfg;
    cfg.id = "ocr_sink_test";
    cfg.input_url = "test_data/ocr_text_loop.mp4";
    cfg.enable_preview = false;
    ModelConfig m;
    m.name = "ocr";
    m.type = "ocr";
    m.backend = "ort";
    m.device = "cpu";
    m.path = "test_data/test_models/onnx/ocr/ppocrv6_tiny/det_infer.onnx";
    m.cls_path = "test_data/test_models/onnx/ocr/ppocrv6_tiny/cls_infer.onnx";
    m.rec_path = "test_data/test_models/onnx/ocr/ppocrv6_tiny/rec_infer.onnx";
    m.dict_path = "test_data/ppocrv6_tiny_dict.txt";
    cfg.models.push_back(m);

    Pipeline pipe(cfg);
    std::atomic<int> text_boxes{0};
    pipe.set_detection_sink([&](const std::vector<DetectionBox>& boxes, int w, int h, double) {
        REQUIRE(w > 0);
        REQUIRE(h > 0);
        for (const auto& b : boxes)
            if (!b.text.empty()) ++text_boxes;
    });
    REQUIRE(pipe.start());
    // 有界等待（模型加载 + 多帧推理，上限 30s）
    for (int i = 0; i < 600 && text_boxes.load() == 0; ++i)
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
    pipe.stop();
    REQUIRE(text_boxes.load() > 0);
}

// LPR pipeline 送达 sink 的集成用例：驱动真实 Pipeline（视频/多帧路径），断言检测
// sink 收到带非空 text（车牌号）的 DetectionBox。单图 infer 已由 test_inference_lpr 覆盖，
// 本用例专门覆盖视频帧（解码为 CPU NV12）路径：回归 det 预处理把 NV12 的 Y 平面当
// packed BGR 越界读取导致进程静默崩溃（0xC0000005）。
TEST_CASE("Pipeline lpr delivers plate text to sink",
          "[agent][pipeline][integration]") {
    const char* kLprVideo = "test_data/lpr_plate_loop.mp4";
    const char* kLprImg = "test_data/test_images/test_lpr_detection.jpg";
    const char* kDet = "test_data/test_models/onnx/yolov5plate.onnx";
    const char* kRec = "test_data/test_models/onnx/plate_recognition_color.onnx";
    if (!std::ifstream(kDet).good() || !std::ifstream(kRec).good() ||
        !ensure_lpr_video(kLprImg, kLprVideo))
        SKIP("LPR test data/ffmpeg absent");

    TaskConfig cfg;
    cfg.id = "lpr_sink_test";
    cfg.input_url = kLprVideo;
    cfg.enable_preview = false;
    ModelConfig m;
    m.name = "lpr";
    m.type = "lpr";
    m.backend = "ort";
    m.device = "cpu";
    m.path = kDet;       // 车牌检测模型
    m.rec_path = kRec;   // 车牌识别模型
    cfg.models.push_back(m);

    Pipeline pipe(cfg);
    std::atomic<int> plate_boxes{0};
    pipe.set_detection_sink([&](const std::vector<DetectionBox>& boxes, int w, int h, double) {
        REQUIRE(w > 0);
        REQUIRE(h > 0);
        for (const auto& b : boxes)
            if (!b.text.empty()) ++plate_boxes;
    });
    REQUIRE(pipe.start());
    // 有界等待（模型加载 + 多帧推理，上限 30s）
    for (int i = 0; i < 600 && plate_boxes.load() == 0; ++i)
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
    pipe.stop();
    REQUIRE(plate_boxes.load() > 0);
}

TEST_CASE("Pipeline without sink unaffected", "[agent][pipeline]") {
    TaskConfig cfg;
    cfg.id = "no_sink";
    cfg.input_url = "E:/nonexistent.mp4";
    cfg.output_url = "x.flv";
    Pipeline pipe(cfg);
    REQUIRE(pipe.start());
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    pipe.stop();
    REQUIRE_FALSE(pipe.is_running());
}
