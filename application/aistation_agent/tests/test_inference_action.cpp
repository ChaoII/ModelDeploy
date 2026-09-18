// B5 家族接线用例：action（PP-TSMv2 视频动作分类，16 帧）、action_skeleton（ST-GCN 骨架动作，
// 由 pose 关键点驱动，窗口取模型声明的 T=300）。
// 1) 类型名归一化（action/action_skeleton 别名）与能力上报
// 2) ConfigAdapter：action 单模型（topk）；action_skeleton 复合（STGCN+pose）与同任务 pose 复用；
//    缺 pose 模型显式失败
// 3) 预处理测试缝：ActionCls::assemble_clip（均匀采样 + 短边 256/中心裁剪 + ImageNet 归一化）、
//    ActionSkeleton::coco17_to_ntu25 / assemble_skeleton（恒等预处理 + [1,C,T,V,M] 形状）
// 4) 真机推理：action 滚动缓冲有界 + cadence；真实片段（Pipeline 解码 mp4）产出真实类别标签
// 5) action_skeleton：pose 关键点 → ST-GCN 真实标签；骨架缓冲有界；真实人视频端到端
// 6) 单帧家族回归：classification 不触碰时序缓冲
// 7) max_embeddings：配置顶层解析经 EventMeta 真正约束嵌入对象数（reid 与 face_rec 共用通路）
#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdio>
#include <fstream>
#include <set>
#include <string>
#include <thread>
#include <vector>

#include <opencv2/core.hpp>

#include "capability.hpp"
#include "config_adapter.hpp"
#include "csrc/core/tensor.h"
#include "csrc/video/video_decoder.h"
#include "event_bus.hpp"
#include "inference_engine.hpp"
#include "nlohmann/json.hpp"
#include "pipeline.hpp"

using nlohmann::json;
using Catch::Approx;
using modeldeploy::Tensor;
using namespace modeldeploy::vision;
using modeldeploy::vision::action::ActionCls;
using modeldeploy::vision::action::ActionSkeleton;
using modeldeploy::vision::action::KeyPointSeq;

namespace {
// 权重/素材（test_data 为 gitignore 的本地资产，缺失则 SKIP）
const char* kActionModel = "test_data/test_models/onnx/action/PPTSMv2_k400_16f.onnx";
const char* kSkeletonModel = "test_data/test_models/onnx/action_skeleton/STGCN_ntucs.onnx";
const char* kPoseModel = "test_data/test_models/onnx/yolo11n/yolo11n-pose_nms.onnx";
const char* kClsModel = "test_data/test_models/onnx/yolo11n/yolo11n-cls.onnx";
const char* kPersonImg = "test_data/test_images/test_person.jpg";
// 测试用小视频：由人像图生成的 640 宽循环片（14s@30fps=420 帧），规避 1080p 硬解在部分机器上
// 的 pkt_timebase 异常，且足够填满 ST-GCN 的 300 帧窗口。
const char* kPersonLoopVideo = "test_data/action_person_loop.mp4";
const char* kK400Labels = "application/demo_labels/kinetics400.txt";
const char* kNtu60Labels = "application/demo_labels/ntu60.txt";

bool file_nonempty(const char* path) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    return f.good() && f.tellg() > 0;
}

bool exists(const char* p) {
    std::ifstream f(p);
    return f.good();
}

// 生成小尺寸循环人像视频（供时序 Pipeline 集成测试；ffmpeg 缺失则返回 false）
bool ensure_person_loop_video(const char* img, const char* out) {
    if (file_nonempty(out)) return true;
    if (!file_nonempty(img)) return false;
    const std::string cmd = std::string("ffmpeg -y -loglevel error -loop 1 -i \"") + img +
                            "\" -vf \"scale=640:trunc(ih*640/iw/2)*2\" -t 14 -r 30 "
                            "-pix_fmt yuv420p \"" + out + "\"";
    if (std::system(cmd.c_str()) != 0) return false;
    return file_nonempty(out);
}

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

ModelConfig make_action_model(const std::vector<std::string>& labels) {
    ModelConfig m;
    m.name = "action";
    m.type = "action";
    m.backend = "ort";
    m.device = "cpu";
    m.path = kActionModel;
    m.labels = labels;
    m.top_k = 5;
    return m;
}

ModelConfig make_skeleton_model(const std::vector<std::string>& labels) {
    ModelConfig m;
    m.name = "action_skeleton";
    m.type = "action_skeleton";
    m.backend = "ort";
    m.device = "cpu";
    m.path = kSkeletonModel;
    m.rec_path = kPoseModel;
    m.input_size = {640, 640};
    m.labels = labels;
    m.top_k = 5;
    return m;
}

// 等待 Pipeline 事件（最多 timeout_ms）
size_t wait_events(std::vector<DetectionEvent>* got, int timeout_ms) {
    const int step = 50;
    for (int waited = 0; waited < timeout_ms && got->empty(); waited += step)
        std::this_thread::sleep_for(std::chrono::milliseconds(step));
    return got->size();
}

// 同步解码一段真实视频的前 n 帧（SDK VideoDecoder，绕过 Pipeline 的有界丢帧队列，
// 使 ST-GCN 的 300 帧窗口能在测试中凑满）。软解以避免部分机器的硬解异常。
std::vector<ImageData> decode_video_frames(const char* path, int n) {
    std::vector<ImageData> out;
    modeldeploy::video::VideoDecoderConfig dc;
    dc.hw_accel = modeldeploy::video::HwAccel::None;
    std::string err;
    auto dec = modeldeploy::video::VideoDecoder::create(dc, &err);
    if (!dec || !dec->open(path, &err)) return out;
    for (int i = 0; i < n; ++i) {
        modeldeploy::video::VideoFrame f;
        if (!dec->read_one_frame(&f, &err)) break;
        out.push_back(f.image.clone());
    }
    dec->close();
    return out;
}
}  // namespace

// ── 类型名归一化 ──
TEST_CASE("normalize_model_type accepts action/action_skeleton aliases", "[agent][config]") {
    REQUIRE(normalize_model_type("action") == "action");
    REQUIRE(normalize_model_type("action_cls") == "action");
    REQUIRE(normalize_model_type("video_action") == "action");
    REQUIRE(normalize_model_type("action_recognition") == "action");
    REQUIRE(normalize_model_type("pptsm") == "action");
    REQUIRE(normalize_model_type("pptsm_v2") == "action");
    REQUIRE(normalize_model_type("tsn") == "action");
    REQUIRE(normalize_model_type("action_skeleton") == "action_skeleton");
    REQUIRE(normalize_model_type("skeleton_action") == "action_skeleton");
    REQUIRE(normalize_model_type("stgcn") == "action_skeleton");
    REQUIRE(normalize_model_type("st_gcn") == "action_skeleton");
    REQUIRE(normalize_model_type("skeleton_gcn") == "action_skeleton");
}

TEST_CASE("detect_capabilities advertises action/action_skeleton families", "[agent][capability]") {
    auto c = detect_capabilities(8);
    std::set<std::string> fams;
    for (const auto& f : c["model_families"]) fams.insert(f.get<std::string>());
    REQUIRE(fams.count("action") == 1);
    REQUIRE(fams.count("action_skeleton") == 1);
}

// ── ConfigAdapter：action 单模型（topk） ──
TEST_CASE("ConfigAdapter parses action single model with topk", "[agent][config]") {
    json j = json::parse(R"({
      "task_id": 74,
      "camera": {"id": 1, "name": "c", "url": "test_data/test_video60.mp4"},
      "models": [
        {"name": "action", "type": "action", "backend": "ort", "device": "cpu",
         "url": "test_data/test_models/onnx/action/PPTSMv2_k400_16f.onnx", "topk": 7}
      ],
      "events": {"transport": "http", "http": {"url": "http://x/cb", "token": "t"}}
    })");
    ConfigAdapter adapter;
    AdaptedTask t;
    std::string err;
    REQUIRE(adapter.from_json(j, &t, &err));
    REQUIRE(t.sdk.models.size() == 1);
    REQUIRE(t.sdk.models[0].type == "action");
    REQUIRE(t.sdk.models[0].path == kActionModel);
    REQUIRE(t.sdk.models[0].top_k == 7);
}

// ── ConfigAdapter：action_skeleton 复用同任务 pose 条目 ──
TEST_CASE("ConfigAdapter action_skeleton reuses sibling pose model", "[agent][config]") {
    json j = json::parse(R"({
      "task_id": 75,
      "camera": {"id": 1, "name": "c", "url": "test_data/test_video60.mp4"},
      "models": [
        {"name": "pose", "type": "pose", "backend": "ort", "device": "cpu",
         "url": "test_data/test_models/onnx/yolo11n/yolo11n-pose_nms.onnx"},
        {"name": "action_skeleton", "type": "action_skeleton", "backend": "ort", "device": "cpu",
         "url": "test_data/test_models/onnx/action_skeleton/STGCN_ntucs.onnx"}
      ],
      "events": {"transport": "http", "http": {"url": "http://x/cb", "token": "t"}}
    })");
    ConfigAdapter adapter;
    AdaptedTask t;
    std::string err;
    REQUIRE(adapter.from_json(j, &t, &err));
    REQUIRE(t.sdk.models.size() == 2);
    REQUIRE(t.sdk.models[1].type == "action_skeleton");
    REQUIRE(t.sdk.models[1].path == kSkeletonModel);   // 主模型 = STGCN
    REQUIRE(t.sdk.models[1].rec_path == kPoseModel);   // 关键点模型 = 同任务 pose
}

// ── ConfigAdapter：action_skeleton 显式 pose / 缺 pose 失败 ──
TEST_CASE("ConfigAdapter action_skeleton explicit pose and missing-pose rejection", "[agent][config]") {
    ConfigAdapter adapter;
    {
        json j = json::parse(R"({
          "task_id": 76,
          "camera": {"id": 1, "name": "c", "url": "test_data/test_video60.mp4"},
          "models": [
            {"name": "action_skeleton", "type": "action_skeleton", "backend": "ort", "device": "cpu",
             "url": "test_data/test_models/onnx/action_skeleton/STGCN_ntucs.onnx",
             "pose_url": "test_data/test_models/onnx/yolo11n/yolo11n-pose_nms.onnx"}
          ],
          "events": {"transport": "http", "http": {"url": "http://x/cb", "token": "t"}}
        })");
        AdaptedTask t;
        std::string err;
        REQUIRE(adapter.from_json(j, &t, &err));
        REQUIRE(t.sdk.models[0].path == kSkeletonModel);
        REQUIRE(t.sdk.models[0].rec_path == kPoseModel);
    }
    {
        json j = json::parse(R"({
          "task_id": 77,
          "camera": {"id": 1, "name": "c", "url": "test_data/test_video60.mp4"},
          "models": [
            {"name": "action_skeleton", "type": "action_skeleton", "backend": "ort", "device": "cpu",
             "url": "test_data/test_models/onnx/action_skeleton/STGCN_ntucs.onnx"}
          ],
          "events": {"transport": "http", "http": {"url": "http://x/cb", "token": "t"}}
        })");
        AdaptedTask t;
        std::string err;
        REQUIRE_FALSE(adapter.from_json(j, &t, &err));
        REQUIRE_FALSE(err.empty());
    }
}

// ── 预处理测试缝：ActionCls::assemble_clip ──
TEST_CASE("ActionCls::assemble_clip samples uniformly and normalizes like PaddleVideo",
          "[agent][model]") {
    const int H = 8, W = 8;
    auto solid = [&](int b, int g, int r) {
        ImageData img(W, H, MdImageType::PKG_BGR_U8);
        cv::Mat mat;
        REQUIRE(img.asMat(&mat));
        mat.setTo(cv::Scalar(b, g, r));
        return img;
    };

    std::vector<ImageData> frames;
    for (int t = 0; t < 4; ++t) frames.push_back(solid(10, 20, 30));
    Tensor out;
    REQUIRE(ActionCls::assemble_clip(frames, 4, H, W, &out));
    REQUIRE(out.shape().size() == 5);
    REQUIRE(out.shape()[0] == 1);
    REQUIRE(out.shape()[1] == 4);
    REQUIRE(out.shape()[2] == 3);
    REQUIRE(out.shape()[3] == H);
    REQUIRE(out.shape()[4] == W);

    const float* p = static_cast<const float*>(out.data());
    // PKG_BGR_U8 → RGB：通道 0=R=30 / 1=G=20 / 2=B=10；ImageNet 归一化
    const float exp_r = (30.f / 255.f - 0.485f) / 0.229f;
    const float exp_g = (20.f / 255.f - 0.456f) / 0.224f;
    const float exp_b = (10.f / 255.f - 0.406f) / 0.225f;
    REQUIRE(p[0] == Approx(exp_r).margin(1e-5f));
    REQUIRE(p[static_cast<size_t>(H) * W] == Approx(exp_g).margin(1e-5f));
    REQUIRE(p[2 * static_cast<size_t>(H) * W] == Approx(exp_b).margin(1e-5f));

    // 均匀采样：K=4, T=2 → 取下标 0 与 2 的帧（不同颜色）
    std::vector<ImageData> varied;
    varied.push_back(solid(0, 0, 255));    // R=255
    varied.push_back(solid(0, 0, 0));
    varied.push_back(solid(0, 255, 0));    // R=0, G=255
    varied.push_back(solid(0, 0, 0));
    Tensor out2;
    REQUIRE(ActionCls::assemble_clip(varied, 2, H, W, &out2));
    const float* q = static_cast<const float*>(out2.data());
    REQUIRE(q[0] == Approx((255.f / 255.f - 0.485f) / 0.229f).margin(1e-5f));  // 帧0
    const size_t frame_stride = 3 * static_cast<size_t>(H) * W;
    REQUIRE(q[frame_stride + static_cast<size_t>(H) * W] ==
            Approx((255.f / 255.f - 0.456f) / 0.224f).margin(1e-5f));          // 帧2 G=255

    // 空输入 / 非法参数
    Tensor out3;
    REQUIRE_FALSE(ActionCls::assemble_clip({}, 4, H, W, &out3));
    REQUIRE_FALSE(ActionCls::assemble_clip(frames, 0, H, W, &out3));
}

// ── 预处理测试缝：ActionSkeleton ──
TEST_CASE("ActionSkeleton maps COCO17 to NTU25 and assembles identity tensor", "[agent][model]") {
    std::vector<Point3f> coco(17);
    for (int i = 0; i < 17; ++i) coco[static_cast<size_t>(i)] = Point3f(100.f + i, 200.f + i, 0.9f);
    auto ntu = ActionSkeleton::coco17_to_ntu25(coco);
    REQUIRE(ntu.size() == 25);
    REQUIRE(ntu[3].x == Approx(coco[0].x));   // head ← nose
    REQUIRE(ntu[4].x == Approx(coco[5].x));   // left shoulder
    REQUIRE(ntu[6].x == Approx(coco[9].x));   // left wrist
    REQUIRE(ntu[0].x == Approx((coco[11].x + coco[12].x) * 0.5f));   // spine base = 髋中点
    REQUIRE(ActionSkeleton::coco17_to_ntu25({}).size() == 25);       // 空输入不崩溃

    KeyPointSeq seq;
    for (int t = 0; t < 3; ++t) {
        std::vector<Point3f> fr(50);                       // M=2 人 × V=25 点
        fr[4] = Point3f(11.f, 12.f, 0.8f);                 // person0 joint4
        fr[25 + 4] = Point3f(21.f, 22.f, 0.7f);            // person1 joint4
        seq.frames.push_back(fr);
    }
    Tensor out;
    REQUIRE(ActionSkeleton::assemble_skeleton(seq, 3, 3, 25, 2, &out));
    REQUIRE(out.shape().size() == 5);
    REQUIRE(out.shape()[0] == 1);
    REQUIRE(out.shape()[1] == 3);   // C
    REQUIRE(out.shape()[2] == 3);   // T
    REQUIRE(out.shape()[3] == 25);  // V
    REQUIRE(out.shape()[4] == 2);   // M
    const float* p = static_cast<const float*>(out.data());
    auto at = [&](int c, int t, int v, int m) {
        return p[((static_cast<size_t>(c) * 3 + static_cast<size_t>(t)) * 25 +
                  static_cast<size_t>(v)) * 2 + static_cast<size_t>(m)];
    };
    // 恒等预处理：原样写入 x/y/score
    REQUIRE(at(0, 0, 4, 0) == Approx(11.f));
    REQUIRE(at(1, 0, 4, 0) == Approx(12.f));
    REQUIRE(at(2, 0, 4, 0) == Approx(0.8f));
    REQUIRE(at(0, 0, 4, 1) == Approx(21.f));
    REQUIRE(at(2, 0, 4, 1) == Approx(0.7f));

    // 帧数不足 → 显式失败
    KeyPointSeq shortseq;
    shortseq.frames.push_back({});
    Tensor o2;
    REQUIRE_FALSE(ActionSkeleton::assemble_skeleton(shortseq, 3, 3, 25, 2, &o2));
}

// ── 真机推理：action 真实图像重复片段 → 真实类别 + 缓冲有界 + cadence ──
TEST_CASE("InferenceEngine action yields real class with bounded buffer and cadence",
          "[agent][model]") {
    if (!exists(kActionModel) || !exists(kPersonImg)) SKIP("action model/image absent");
    auto labels = load_labels(kK400Labels);
    if (labels.size() != 400) SKIP("kinetics-400 labels absent");
    InferenceEngine eng;
    REQUIRE(eng.load(make_action_model(labels)));
    auto img = ImageData::imread(kPersonImg);
    REQUIRE(!img.empty());

    InferResult r;
    for (int i = 0; i < 40; ++i) {
        InferResult tmp;
        REQUIRE(eng.infer(img, &tmp));
        if (i == 39) r = tmp;
    }
    // 缓冲有界：<= 2*T = 32
    REQUIRE(eng.action_buffer_size() == 32);
    // cadence：40 帧每 16 帧一次 → 2 次推理
    REQUIRE(eng.action_run_count() == 2);
    REQUIRE(r.type == "action");
    REQUIRE_FALSE(r.boxes.empty());
    const auto& b = r.boxes[0];
    REQUIRE(b.x == 0.f);
    REQUIRE(b.y == 0.f);
    REQUIRE(b.w == static_cast<float>(img.width()));
    REQUIRE(b.h == static_cast<float>(img.height()));
    REQUIRE_FALSE(b.label_name.empty());
    REQUIRE_FALSE(b.attributes.empty());
    REQUIRE(b.attributes[0].first == b.label_name);
    std::printf("[smoke] action: label=%s id=%d score=%.6f buffer=%zu runs=%lld\n",
                b.label_name.c_str(), b.label_id, static_cast<double>(b.score),
                eng.action_buffer_size(),
                static_cast<long long>(eng.action_run_count()));
}

// ── 真机推理：action_skeleton 由真实人视频的 pose 关键点驱动 → 真实 NTU 类别 + 骨架缓冲有界 ──
TEST_CASE("InferenceEngine action_skeleton classifies pose keypoints from a person video",
          "[agent][model]") {
    if (!exists(kSkeletonModel) || !exists(kPoseModel)) SKIP("action_skeleton/pose model absent");
    if (!ensure_person_loop_video(kPersonImg, kPersonLoopVideo)) SKIP("person loop video absent");
    auto labels = load_labels(kNtu60Labels);
    if (labels.size() != 60) SKIP("ntu60 labels absent");
    auto frames = decode_video_frames(kPersonLoopVideo, 305);
    if (frames.size() < 300) SKIP("video decode failed/too short");

    InferenceEngine eng;
    REQUIRE(eng.load(make_skeleton_model(labels)));
    InferResult r;
    for (size_t i = 0; i < frames.size(); ++i) {
        InferResult tmp;
        REQUIRE(eng.infer(frames[i], &tmp));
        if (i + 1 == frames.size()) r = tmp;
    }
    // 缓冲有界：<= T = 300
    REQUIRE(eng.action_skeleton_buffer_size() <= 300);
    REQUIRE(eng.action_skeleton_run_count() >= 1);   // 300 帧骨架就绪后至少推理一次
    REQUIRE(r.type == "action_skeleton");
    REQUIRE_FALSE(r.boxes.empty());
    const auto& b = r.boxes[0];
    REQUIRE_FALSE(b.label_name.empty());
    REQUIRE_FALSE(b.attributes.empty());
    REQUIRE(b.attributes[0].first == b.label_name);
    std::printf("[smoke] action_skeleton: label=%s id=%d score=%.6f buffer=%zu runs=%lld\n",
                b.label_name.c_str(), b.label_id, static_cast<double>(b.score),
                eng.action_skeleton_buffer_size(),
                static_cast<long long>(eng.action_skeleton_run_count()));
}

// ── 集成（真实 mp4）：action 经 Pipeline 解码 → sink → 事件，label/attributes 落地 ──
TEST_CASE("Pipeline action yields real class on a real video", "[agent][pipeline][integration]") {
    if (!exists(kActionModel)) SKIP("action model absent");
    if (!ensure_person_loop_video(kPersonImg, kPersonLoopVideo)) SKIP("person loop video absent");
    auto labels = load_labels(kK400Labels);
    if (labels.size() != 400) SKIP("kinetics-400 labels absent");

    TaskConfig cfg;
    cfg.id = "action_sink_test";
    cfg.input_url = kPersonLoopVideo;
    cfg.decoder.hw_accel = "none";   // 强制软解，规避硬解 pkt_timebase 异常
    cfg.enable_preview = false;
    cfg.models.push_back(make_action_model(labels));

    EventBus bus;
    std::vector<DetectionEvent> got;
    bus.set_sink([&](const DetectionEvent& e) { got.push_back(e); });
    EventMeta meta;
    meta.edge_code = "edge-01";
    meta.camera_id = 1;
    meta.task_id_num = 74;
    meta.alarm_interval_sec = 0;
    bus.register_task(cfg.id, meta);

    Pipeline pipe(cfg);
    pipe.set_detection_sink([&](const std::vector<DetectionBox>& boxes, int w, int h, double lat) {
        bus.on_detections(cfg.id, boxes, w, h, lat);
    });
    REQUIRE(pipe.start());
    wait_events(&got, 30000);
    pipe.stop();

    REQUIRE_FALSE(got.empty());
    const auto j = got[0].to_json();
    REQUIRE_FALSE(j["objects"].empty());
    const auto& obj = j["objects"][0];
    REQUIRE_FALSE(obj.value("label", "").empty());
    REQUIRE(obj["bbox"]["x"].get<float>() == Approx(0.f));
    REQUIRE(obj["bbox"]["y"].get<float>() == Approx(0.f));
    REQUIRE(obj["bbox"]["width"].get<float>() == Approx(1.f));
    REQUIRE(obj["bbox"]["height"].get<float>() == Approx(1.f));
    REQUIRE_FALSE(obj["attributes"].empty());
    std::printf("[smoke] action pipeline: label=%s events=%zu\n",
                obj["label"].get<std::string>().c_str(), got.size());
}

// ── 单帧家族回归：classification 不触碰时序缓冲 ──
TEST_CASE("single-frame families are unaffected by temporal action buffers", "[agent][model]") {
    if (!exists(kClsModel) || !exists(kPersonImg)) SKIP("classification model/image absent");
    ModelConfig m;
    m.name = "cls";
    m.type = "classification";
    m.backend = "ort";
    m.device = "cpu";
    m.path = kClsModel;
    m.labels.reserve(1000);
    for (int i = 0; i < 1000; ++i) m.labels.push_back("class" + std::to_string(i));
    InferenceEngine eng;
    REQUIRE(eng.load(m));
    auto img = ImageData::imread(kPersonImg);
    REQUIRE(!img.empty());
    InferResult r;
    REQUIRE(eng.infer(img, &r));
    REQUIRE(r.type == "classification");
    REQUIRE_FALSE(r.boxes.empty());
    REQUIRE(eng.action_buffer_size() == 0);
    REQUIRE(eng.action_skeleton_buffer_size() == 0);
    REQUIRE(eng.action_run_count() == 0);
    REQUIRE(eng.action_skeleton_run_count() == 0);
}

// ── max_embeddings：配置顶层解析 → EventMeta → 嵌入对象数上限（reid 走同一通路） ──
TEST_CASE("max_embeddings parsed from config caps embeddings on reid detections",
          "[agent][config][events]") {
    json j = json::parse(R"({
      "task_id": 91,
      "camera": {"id": 1, "name": "c", "url": "test_data/test_video60.mp4"},
      "max_embeddings": 1,
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
    REQUIRE(t.max_embeddings == 1);   // 配置解析确实携带该上限

    // reid 的 objects[].embedding 与 face_rec 共用 EventBus 嵌入通路，无模型类型门控
    EventBus bus;
    std::vector<DetectionEvent> got;
    bus.set_sink([&](const DetectionEvent& e) { got.push_back(e); });
    EventMeta meta;
    meta.task_id_num = 91;
    meta.alarm_interval_sec = 0;
    meta.max_embeddings = t.max_embeddings;   // 与 agent_runtime 的传递口径一致
    bus.register_task("t", meta);

    std::vector<DetectionBox> dets;
    for (int i = 0; i < 3; ++i) {
        DetectionBox b;
        b.x = static_cast<float>(i) * 10.f;
        b.y = 0.f;
        b.w = 10.f;
        b.h = 10.f;
        b.score = 0.9f - 0.1f * static_cast<float>(i);
        b.label_id = 0;
        b.label_name = "person";
        b.embedding = std::vector<float>(512, 0.1f);
        dets.push_back(std::move(b));
    }
    bus.on_detections("t", dets, 100, 100, 1.0);
    REQUIRE(got.size() == 1);
    const auto e = got[0].to_json();
    size_t with_emb = 0;
    for (const auto& o : e["objects"])
        if (o.contains("embedding")) ++with_emb;
    REQUIRE(with_emb >= 1);
    REQUIRE(with_emb <= 1);   // max_embeddings=1 生效
}
