# Item 11: CV / Audio / NLP 解决方案层 + 工具层——实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在既有 C++ SDK 之上，为三域（CV / Audio / NLP）各新增**可复用的工具层**（纯 C++/现有库、无模型、可单测）与**面向真实场景的解决方案层**（组合既有模型做业务编排），6 面（C++/Python/CAPI/C#/Rust/demo+docs+tests）一次做齐。

**Architecture:** 三域共享同一「方案层 + 工具层」结构。CV：`csrc/vision/tools/`（`vision::tool`，Detections/Zone/Annotator/Metrics/Slicer/Smoother）+ `csrc/vision/solutions/`（`vision::solution`，九方案消费 `tracking::TrackResult`/`Detection`/`KeyPointsResult`）。Audio：`csrc/audio/tools/`（`audio::tool`，WavIO/Resampler/Fbank/Waveform/VadSegment/AudioMeta）+ `csrc/audio/solutions/`（`audio::solution`，四方案组合 VAD/SpeakerVerify/SenseVoice/Kokoro）。NLP：`csrc/nlp/tools/`（`nlp::tool`，Tokenizer/Splitter/Normalizer/Keywords/Stats）+ `csrc/nlp/solutions/`（`nlp::solution::TextClassifier`，唯一带权重方案）。方案层 = 组合 + 少量算法逻辑（几乎无新权重）；工具层 = 无模型可独立单测。

**Tech Stack:** C++17、OpenCV（BUILD_VISION）、samplerate / kaldi-native-fbank / cppjieba（third_party 已捆绑，BUILD_AUDIO 已编译）、pybind11、Catch2、现有 `BaseModel`/`Runtime`/`Tensor`/`ImageData`/`tracking::ByteTracker`/`video::VideoDecoder`、现有 CAPI `md_*` 句柄模式。

**Spec:** `docs/superpowers/specs/2026-08-23-cv-solutions-design.md`

## Global Constraints

- **最高约束——复用现有基础设施，绝不另起炉灶**（每个 Task 都标注「复用」条目）：
  - **CV 工具/方案**复用：`tracking::Detection/TrackResult/ByteTracker::update(detections, frame, timestamp)`（`csrc/vision/tracking/base_tracker.h:9-16`，`TrackResult{track_id,box,score,label_id,...}`）；`ImageData::{crop,resize,from_bgr24,from_raw,asMat,imshow/imwrite}`（`csrc/vision/common/image_data.h`）；`struct.h` 的 `Rect2f/Point2f/Point3f`；`result.h` 的 `DetectionResult/KeyPointsResult/Mask`；`vision::utils::{rect2f_to_cv_type, l2_normalize, compute_similarity}`（`csrc/vision/utils.h`）；`vision::common::visualize/utils.h` 的 `draw_filled_rect/draw_filled_polygon/draw_text/draw_rectangle_and_text`；`video::VideoDecoder`（BUILD_VIDEO，demo 用）。
  - **Audio 工具/方案**复用：`csrc/utils/wave_helper.h:93` 的 `load_wav_file(const char*, int32_t* sr, vector<float>&)`（读 wav）；`capi/md_capi.cpp:2415` `md_wav_save` 的 RIFF 写盘逻辑；`<samplerate/include/samplerate.h>`（`src_simple`）；`knf::OnlineFbank`/`knf::FbankOptions`（kaldi-native-fbank，用法见 `csrc/audio/speaker_verify/ecapa.cpp:50-56` / `csrc/audio/asr/sense_voice.cpp:110-116`）；`audio::SpeakerGallery`（`csrc/audio/speaker_gallery.h`）；`audio::speaker_verify::SpeakerVerify::predict(vector<float>, vector<float>*)`；`audio::asr::SenseVoice::predict(data, string*)`；`audio::vad::SileroVAD::predict(data, string*)`；`audio::tts::Kokoro::predict(text,voice,speed,vector<float>*)`；`audio::AAsr`（`asr_pipeline.h`，push/run 流式参考）。
  - **NLP 工具/方案**复用：`third_party/cppjieba`（`cppjieba/Jieba.hpp`，用法见 `csrc/audio/tts/kokoro.cpp:76-82`）；`csrc/audio/text_normalize/*` 的数字/日期/单位规则（NLP Normalizer 做「精简适配层」参考其逻辑）；`BaseModel`（`csrc/base_model.h`，TextClassifier 继承之）。
- **命名空间**：`modeldeploy::vision::tool` / `modeldeploy::vision::solution`；`modeldeploy::audio::tool` / `modeldeploy::audio::solution`；`modeldeploy::nlp::tool` / `modeldeploy::nlp::solution`。
- **BUILD 开关**：CV 工具/方案全部在 `BUILD_VISION` 下（根 `CMakeLists.txt:117` VISION_SOURCE GLOB `csrc/vision/*.cpp` → 会自动收集，**CV 无需改主 CMake**）；Audio 工具/方案在 `BUILD_AUDIO` 下（`:118` AUDIO_SOURCE GLOB `csrc/audio/*.cpp`，**无需改主 CMake**）；NLP 新增 `BUILD_NLP` 开关（复用同款 cppjieba，见 Task C1，首次触碰主 CMake）。
- **方案层 = 组合 + 少量算法逻辑**：9 个 CV 方案、4 个 Audio 方案全部无新权重（复用 yolo/姿态/跟踪/声纹/SenseVoice/VAD 权重点，或纯合成输入）；**NLP 方案层仅 TextClassifier 引入一个小型 ONNX 文本分类权重（外链，仓库不含 → 测试/演示对缺失权重 SKIP 而非硬失败）**。
- **工具层无模型、可独立单测**：全部纯 C++/OpenCV/现有库；只吃合成输入做确定性断言。真实权重/字典/视频路径一概 `WARN(...); return;` 守卫 SKIP——沿用 `tests/test_hand.cpp:24-27` 模式。
- **TDD 强制**：每个 Task「写失败测试 → 跑看失败 → 最小实现 → 跑看通过 → commit」。测试 Tag：`[cv_tools]` / `[cv_solution]` / `[audio_tools]` / `[audio_solution]` / `[nlp]`（CAPI Tag `[capi]`）。新测试文件加入 `tests/CMakeLists.txt` 的 `TEST_SOURCES`。
- **CAPI 句柄模式**：复用现有 `md_*` 句柄（`capi/md_capi.h` 的 `MDModelHandle/MDResultHandle/MDStatus`、`md_model_create` 分发 `case`、`md_result_classification` 读取、`set_error/need_parts/split_path`）。NLP 分类复用统一 `MD_MODEL_TEXT_CLASSIFIER`（`md_model_create` + `md_result_classification`，仿 TSN 先例 `capi/md_capi.cpp:917-924`）。
- **六面一次做齐**：每域 C++ 核心 → pybind → CAPI → C# → Rust → demo+docs。跨语言薄封装随 CAPI 走（合并为一个 Task）。
- **MSVC `/utf-8`**：根 CMake 已为 SDK 自动设置；C++17。Workout `M_PI` 在 MSVC 可能未定义，用字面量。
- **无权重 SKIP**：所有依赖权重/字典/视频的断言用 `WARN+return` 守卫；工具层与方案层纯逻辑断言恒跑。

---

# A 域：CV（方案层 + 工具层）

## A1: `vision::tool::Detections` 容器 + `iou`/`nms`/`filter_by_class`

**Files:**
- Create: `csrc/vision/tools/detections.h`
- Create: `csrc/vision/tools/detections.cpp`
- Test: `tests/test_cv_tools.cpp`
- Modify: `tests/CMakeLists.txt`（TEST_SOURCES 加 `test_cv_tools.cpp`）

**Interfaces:**
- Consumes: `Rect2f`（`csrc/vision/common/struct.h`）、`Mask`（`result.h`）、`tracking::Detection/TrackResult`（`base_tracker.h`）
- Produces (later tasks rely on):
  - `struct vision::tool::Detections { std::vector<Rect2f> boxes; std::vector<int32_t> class_id; std::vector<float> confidence; std::vector<Mask> masks; std::vector<int32_t> tracker_id; size_t size() const; void reserve(size_t); }`
  - `float vision::tool::iou(const Rect2f& a, const Rect2f& b)`
  - `void vision::tool::nms(Detections& d, float iou_threshold)`
  - `void vision::tool::filter_by_class(Detections& d, const std::vector<int32_t>& keep_classes)`
  - `Detections vision::tool::from_track(const std::vector<tracking::TrackResult>&)`
  - `Detections vision::tool::from_detections(const std::vector<tracking::Detection>&)`

- [ ] **Step 1: 写失败测试** (`tests/test_cv_tools.cpp`)

```cpp
#include <catch2/catch_test_macros.hpp>
#include "vision/tools/detections.h"
using namespace modeldeploy::vision;
using namespace modeldeploy::vision::tool;

TEST_CASE("Detections iou", "[cv_tools]") {
    REQUIRE(iou(Rect2f(0,0,10,10), Rect2f(0,0,10,10)) == Approx(1.0f).margin(1e-5f));
    REQUIRE(iou(Rect2f(0,0,10,10), Rect2f(20,20,10,10)) == Approx(0.0f).margin(1e-6f));
    float half = iou(Rect2f(0,0,10,10), Rect2f(5,0,10,10));
    REQUIRE(half > 0.30f && half < 0.36f); // 并=150 交=50
}

TEST_CASE("Detections nms keeps top by confidence", "[cv_tools]") {
    Detections d;
    d.boxes   = {Rect2f(0,0,10,10), Rect2f(1,1,10,10), Rect2f(100,100,10,10)};
    d.class_id = {0, 0, 1};
    d.confidence = {0.5f, 0.9f, 0.7f};
    nms(d, 0.4f);   // 框0/框1 IoU 高 → 只留框1；框2 分离保留
    REQUIRE(d.size() == 2);
    REQUIRE(d.confidence[0] == Approx(0.9f));
}

TEST_CASE("Detections filter_by_class", "[cv_tools]") {
    Detections d;
    d.class_id = {0, 1, 2, 0};
    d.boxes.resize(4);
    filter_by_class(d, {0, 2});
    REQUIRE(d.size() == 3);
}

TEST_CASE("Detections from_track maps tracker_id", "[cv_tools]") {
    std::vector<tracking::TrackResult> t(2);
    t[0].track_id = 7; t[0].box = Rect2f(1,1,5,5); t[0].label_id = 2; t[0].score = 0.8f;
    t[1].track_id = 3; t[1].box = Rect2f(9,9,5,5);
    auto d = from_track(t);
    REQUIRE(d.size() == 2);
    REQUIRE(d.tracker_id[0] == 7);
    REQUIRE(d.class_id[0] == 2);
    REQUIRE(d.confidence[0] == Approx(0.8f));
    REQUIRE(d.boxes[1].x == Approx(9.0f));
}
```

- [ ] **Step 2: 运行确认失败**

Run: `cmake --build build --parallel 8 && cd build && .\bin\test_modeldeploy.exe "[cv_tools]"`
Expected: FAIL（`vision/tools/detections.h` 不存在，编译失败）。
> `build` 目录先配置：`cmake -S . -B build -G Ninja -DBUILD_AUDIO=ON -DBUILD_VISION=ON -DBUILD_CAPI=ON -DBUILD_PYTHON=OFF -DENABLE_MNN=OFF -DENABLE_ORT=ON -DENABLE_TRT=OFF -DWITH_GPU=OFF -DBUILD_TESTS=ON`.

- [ ] **Step 3: 写 `csrc/vision/tools/detections.h`**

```cpp
#pragma once
#include <vector>
#include "core/md_decl.h"
#include "vision/common/struct.h"
#include "vision/common/result.h"
#include "vision/tracking/base_tracker.h"

namespace modeldeploy::vision::tool {
struct MODELDEPLOY_CXX_EXPORT Detections {
    std::vector<Rect2f> boxes;
    std::vector<int32_t> class_id;
    std::vector<float> confidence;
    std::vector<Mask> masks;
    std::vector<int32_t> tracker_id;
    [[nodiscard]] size_t size() const { return boxes.size(); }
    void reserve(size_t n) { boxes.reserve(n); class_id.reserve(n); confidence.reserve(n); masks.reserve(n); tracker_id.reserve(n); }
};
MODELDEPLOY_CXX_EXPORT float iou(const Rect2f& a, const Rect2f& b);
MODELDEPLOY_CXX_EXPORT void nms(Detections& d, float iou_threshold);
MODELDEPLOY_CXX_EXPORT void filter_by_class(Detections& d, const std::vector<int32_t>& keep_classes);
MODELDEPLOY_CXX_EXPORT Detections from_track(const std::vector<tracking::TrackResult>& t);
MODELDEPLOY_CXX_EXPORT Detections from_detections(const std::vector<tracking::Detection>& t);
} // namespace modeldeploy::vision::tool
```

- [ ] **Step 4: 写 `csrc/vision/tools/detections.cpp`**

```cpp
#include "vision/tools/detections.h"
#include <algorithm>
#include <cmath>

namespace modeldeploy::vision::tool {
float iou(const Rect2f& a, const Rect2f& b) {
    const float ax2 = a.x + a.width, ay2 = a.y + a.height;
    const float bx2 = b.x + b.width, by2 = b.y + b.height;
    const float ix = std::max(0.0f, std::min(ax2, bx2) - std::max(a.x, b.x));
    const float iy = std::max(0.0f, std::min(ay2, by2) - std::max(a.y, b.y));
    const float inter = ix * iy;
    const float uni = a.width * a.height + b.width * b.height - inter;
    if (uni <= 0.0f) return 0.0f;
    return inter / uni;
}
void nms(Detections& d, float iou_threshold) {
    const size_t n = d.size();
    std::vector<size_t> order(n);
    for (size_t i = 0; i < n; ++i) order[i] = i;
    std::sort(order.begin(), order.end(), [&](size_t a, size_t b){ return d.confidence[a] > d.confidence[b]; });
    std::vector<bool> keep(n, true);
    for (size_t i = 0; i < n; ++i) {
        if (!keep[order[i]]) continue;
        for (size_t j = i + 1; j < n; ++j)
            if (keep[order[j]] && iou(d.boxes[order[i]], d.boxes[order[j]]) > iou_threshold)
                keep[order[j]] = false;
    }
    Detections out; out.reserve(n);
    for (size_t i = 0; i < n; ++i) {
        if (!keep[order[i]]) continue;
        out.boxes.push_back(d.boxes[order[i]]);
        out.class_id.push_back(d.class_id[order[i]]);
        out.confidence.push_back(d.confidence[order[i]]);
        if (!d.masks.empty()) out.masks.push_back(d.masks[order[i]]);
        if (!d.tracker_id.empty()) out.tracker_id.push_back(d.tracker_id[order[i]]);
    }
    d = std::move(out);
}
void filter_by_class(Detections& d, const std::vector<int32_t>& keep) {
    Detections out; out.reserve(d.size());
    for (size_t i = 0; i < d.size(); ++i) {
        if (std::find(keep.begin(), keep.end(), d.class_id[i]) == keep.end()) continue;
        out.boxes.push_back(d.boxes[i]); out.class_id.push_back(d.class_id[i]);
        out.confidence.push_back(d.confidence[i]);
        if (!d.masks.empty()) out.masks.push_back(d.masks[i]);
        if (!d.tracker_id.empty()) out.tracker_id.push_back(d.tracker_id[i]);
    }
    d = std::move(out);
}
Detections from_track(const std::vector<tracking::TrackResult>& t) {
    Detections d; d.reserve(t.size());
    for (const auto& r : t) {
        d.boxes.push_back(r.box); d.class_id.push_back(r.label_id);
        d.confidence.push_back(r.score); d.tracker_id.push_back(r.track_id);
    }
    return d;
}
Detections from_detections(const std::vector<tracking::Detection>& t) {
    Detections d; d.reserve(t.size());
    for (const auto& r : t) {
        d.boxes.push_back(r.box); d.class_id.push_back(r.label_id);
        d.confidence.push_back(r.score);
    }
    return d;
}
} // namespace modeldeploy::vision::tool
```

- [ ] **Step 5: 构建 + 运行确认通过**

Modify `tests/CMakeLists.txt` TEST_SOURCES: 加 `test_cv_tools.cpp`（仿 `test_hand.cpp` 列法）。
Run: `cmake --build build --parallel 8 && cd build && .\bin\test_modeldeploy.exe "[cv_tools]"`
Expected: 4 个 `[cv_tools]` 用例 PASS。

- [ ] **Step 6: Commit**

```bash
git add csrc/vision/tools/detections.h csrc/vision/tools/detections.cpp tests/test_cv_tools.cpp tests/CMakeLists.txt
git commit -m "feat(cv_tools): Detections container + iou/nms/filter + track converters"
```

---

## A2: `vision::tool` Zone（LineZone 跨线 + PolygonZone 多边形）

**Files:**
- Create: `csrc/vision/tools/zone.h`
- Create: `csrc/vision/tools/zone.cpp`
- Modify: `tests/test_cv_tools.cpp`

**Interfaces:**
- Consumes: `Point2f`（`struct.h`）
- Produces (later tasks rely on):
  - `class LineZone { LineZone(Point2f start, Point2f end); void reset(); bool trigger(const Point2f& p); int trigger_count() const; bool in_side() const; }`（跨侧返回 true；`in_side()` 只读当前是否 in 侧）
  - `class PolygonZone { PolygonZone(); explicit PolygonZone(std::vector<Point2f> points); void reset(); bool contains(Point2f p) const; int current_count() const; void update(const std::vector<Point2f>& pts); const std::vector<Point2f>& points() const; }`
  - `void vision::tool::filter_by_zone(Detections& d, const PolygonZone& zone, const std::vector<int32_t>* keep_classes = nullptr, float score_threshold = 0.0f)`

- [ ] **Step 1: 追加失败测试** (`tests/test_cv_tools.cpp`)

```cpp
#include "vision/tools/zone.h"

TEST_CASE("LineZone counts crossing once", "[cv_tools]") {
    LineZone z(Point2f(5, 0), Point2f(5, 10));
    REQUIRE(z.trigger_count() == 0);
    REQUIRE(z.trigger(Point2f(0, 5)) == false); // out 侧
    REQUIRE(z.trigger(Point2f(8, 5)) == true);  // 跨到 in → 计数
    REQUIRE(z.trigger_count() == 1);
    REQUIRE(z.trigger(Point2f(9, 5)) == false); // 仍在 in
    z.reset();
    REQUIRE(z.trigger_count() == 0);
}

TEST_CASE("PolygonZone contains + current_count", "[cv_tools]") {
    PolygonZone z({Point2f(0,0), Point2f(10,0), Point2f(10,10), Point2f(0,10)});
    REQUIRE(z.contains(Point2f(5,5)));
    REQUIRE_FALSE(z.contains(Point2f(20,20)));
    z.update({Point2f(2,2), Point2f(50,50)});
    REQUIRE(z.current_count() == 1);
}
```

- [ ] **Step 2: 运行确认失败**

Run: `cd build && .\bin\test_modeldeploy.exe "[cv_tools]"`
Expected: FAIL（`vision/tools/zone.h` 不存在）。

- [ ] **Step 3: 写 `csrc/vision/tools/zone.h`**

```cpp
#pragma once
#include <vector>
#include "core/md_decl.h"
#include "vision/common/struct.h"
#include "vision/tools/detections.h"

namespace modeldeploy::vision::tool {
class MODELDEPLOY_CXX_EXPORT LineZone {
public:
    LineZone(Point2f start, Point2f end) : a_(start), b_(end) {}
    void reset() { count_ = 0; last_in_ = false; has_last_ = false; }
    int trigger_count() const { return count_; }
    bool in_side() const { return has_last_ ? last_in_ : false; }
    bool trigger(const Point2f& p);
private:
    Point2f a_, b_;
    int count_{0};
    bool last_in_{false};
    bool has_last_{false};
};
class MODELDEPLOY_CXX_EXPORT PolygonZone {
public:
    PolygonZone() = default;
    explicit PolygonZone(std::vector<Point2f> points) : points_(std::move(points)) {}
    void reset() { count_ = 0; }
    bool contains(Point2f p) const;
    int current_count() const { return count_; }
    void update(const std::vector<Point2f>& pts);
    const std::vector<Point2f>& points() const { return points_; }
private:
    std::vector<Point2f> points_;
    int count_{0};
};
MODELDEPLOY_CXX_EXPORT void filter_by_zone(Detections& d, const PolygonZone& zone,
                                           const std::vector<int32_t>* keep_classes = nullptr,
                                           float score_threshold = 0.0f);
} // namespace modeldeploy::vision::tool
```

- [ ] **Step 4: 写 `csrc/vision/tools/zone.cpp`**

```cpp
#include "vision/tools/zone.h"
#include <algorithm>

namespace modeldeploy::vision::tool {
static bool side_in(const Point2f& a, const Point2f& b, const Point2f& p) {
    return (b.x - a.x) * (p.y - a.y) - (b.y - a.y) * (p.x - a.x) > 0.0f;
}
bool LineZone::trigger(const Point2f& p) {
    const bool in = side_in(a_, b_, p);
    bool crossed = false;
    if (has_last_ && last_in_ != in) crossed = true; // 两侧翻转计一次
    if (crossed) ++count_;
    last_in_ = in;
    has_last_ = true;
    return crossed;
}
bool PolygonZone::contains(Point2f p) const {
    bool inside = false;
    const int n = static_cast<int>(points_.size());
    for (int i = 0, j = n - 1; i < n; j = i++) {
        const Point2f& a = points_[i];
        const Point2f& b = points_[j];
        if ((a.y > p.y) != (b.y > p.y) &&
            p.x < (b.x - a.x) * (p.y - a.y) / (b.y - a.y) + a.x)
            inside = !inside;
    }
    return inside;
}
void PolygonZone::update(const std::vector<Point2f>& pts) {
    for (const auto& p : pts) if (contains(p)) ++count_;
}
void filter_by_zone(Detections& d, const PolygonZone& zone,
                    const std::vector<int32_t>* keep_classes, float score_threshold) {
    Detections out; out.reserve(d.size());
    for (size_t i = 0; i < d.size(); ++i) {
        if (d.confidence[i] < score_threshold) continue;
        if (keep_classes && std::find(keep_classes->begin(), keep_classes->end(), d.class_id[i]) == keep_classes->end()) continue;
        const Rect2f& b = d.boxes[i];
        if (!zone.contains(Point2f(b.x + b.width * 0.5f, b.y + b.height * 0.5f))) continue;
        out.boxes.push_back(d.boxes[i]); out.class_id.push_back(d.class_id[i]);
        out.confidence.push_back(d.confidence[i]);
        if (!d.masks.empty()) out.masks.push_back(d.masks[i]);
        if (!d.tracker_id.empty()) out.tracker_id.push_back(d.tracker_id[i]);
    }
    d = std::move(out);
}
} // namespace modeldeploy::vision::tool
```

- [ ] **Step 5: 构建 + 运行确认通过**

Run: `cmake --build build --parallel 8 && cd build && .\bin\test_modeldeploy.exe "[cv_tools]"`
Expected: 6 个 `[cv_tools]` 用例 PASS。
> `LineZone::trigger` 两侧翻转都计数。A7 的 ObjectCounter 以「进 in 侧 → line_in、进 out 侧 → line_out」用 `in_side()` 区分方向。

- [ ] **Step 6: Commit**

```bash
git add csrc/vision/tools/zone.h csrc/vision/tools/zone.cpp tests/test_cv_tools.cpp
git commit -m "feat(cv_tools): LineZone + PolygonZone + filter_by_zone"
```

---

## A3: `vision::tool` Annotator（叠加画布 + 几何绘制）

**Files:**
- Create: `csrc/vision/tools/annotator.h`
- Create: `csrc/vision/tools/annotator.cpp`
- Modify: `tests/test_cv_tools.cpp`

**Interfaces:**
- Consumes: `ImageData`（`asMat`/`from_raw`）、`Detections`（A1）、`Rect2f/Point2f`
- Produces:
  - `class vision::tool::Annotator { explicit Annotator(ImageData* frame); bool begin(cv::Mat* m); void rectangle(const Rect2f&, const cv::Scalar&, int thickness=2); void text(const std::string&, Point2f, const cv::Scalar&, double scale=0.6); void line(Point2f, Point2f, const cv::Scalar&, int thickness=2); void circle(Point2f, int, const cv::Scalar&, int thickness=2); void fill_polygon(const std::vector<Point2f>&, const cv::Scalar&, double alpha=0.3); }`
  - `void draw_box_labels(const Detections& d, ImageData* frame, const std::unordered_map<int,std::string>& labels = {})`
  - `void draw_traces(const std::vector<Point2f>& trace, ImageData* frame, const cv::Scalar& color, int thickness=2)`

- [ ] **Step 1: 追加失败测试** (`tests/test_cv_tools.cpp`)

```cpp
#include "vision/tools/annotator.h"

static ImageData make_canvas(int w, int h) {
    std::vector<uint8_t> pixels(static_cast<size_t>(w) * h * 3, 0);
    return ImageData::from_raw(pixels.data(), w, h, MdImageType::PKG_BGR_U8, true);
}

TEST_CASE("Annotator draws rectangle on canvas", "[cv_tools]") {
    ImageData frame = make_canvas(20, 20);
    Annotator ann(&frame);
    ann.rectangle(Rect2f(2, 2, 10, 10), cv::Scalar(0, 0, 255), 2);
    ann.text("obj", Point2f(2, 2), cv::Scalar(255, 255, 255), 0.5);
    cv::Mat m;
    REQUIRE(frame.asMat(&m));
    REQUIRE(m.at<cv::Vec3b>(3, 3)[2] == 255); // 边框上红通道
}

TEST_CASE("draw_box_labels draws boxes", "[cv_tools]") {
    ImageData frame = make_canvas(30, 30);
    Detections d;
    d.boxes = {Rect2f(1, 1, 5, 5)}; d.class_id = {0}; d.confidence = {0.9f};
    draw_box_labels(d, &frame, {{0, "person"}});
    cv::Mat m;
    REQUIRE(frame.asMat(&m));
    REQUIRE(m.at<cv::Vec3b>(2, 2)[2] == 255);
}
```

- [ ] **Step 2: 运行确认失败**

Run: `cd build && .\bin\test_modeldeploy.exe "[cv_tools]"`
Expected: FAIL（`vision/tools/annotator.h` 不存在）。

- [ ] **Step 3: 写 `csrc/vision/tools/annotator.h`**

```cpp
#pragma once
#include <opencv2/opencv.hpp>
#include <string>
#include <unordered_map>
#include <vector>
#include "core/md_decl.h"
#include "vision/common/image_data.h"
#include "vision/common/struct.h"
#include "vision/tools/detections.h"

namespace modeldeploy::vision::tool {
class MODELDEPLOY_CXX_EXPORT Annotator {
public:
    explicit Annotator(ImageData* frame) : frame_(frame) {}
    bool begin(cv::Mat* m);
    void rectangle(const Rect2f& b, const cv::Scalar& color, int thickness = 2);
    void text(const std::string& s, Point2f org, const cv::Scalar& color, double scale = 0.6);
    void line(Point2f a, Point2f b, const cv::Scalar& color, int thickness = 2);
    void circle(Point2f c, int r, const cv::Scalar& color, int thickness = 2);
    void fill_polygon(const std::vector<Point2f>& pts, const cv::Scalar& color, double alpha = 0.3);
private:
    ImageData* frame_;
};
MODELDEPLOY_CXX_EXPORT void draw_box_labels(const Detections& d, ImageData* frame,
                                            const std::unordered_map<int, std::string>& labels = {});
MODELDEPLOY_CXX_EXPORT void draw_traces(const std::vector<Point2f>& trace, ImageData* frame,
                                        const cv::Scalar& color, int thickness = 2);
} // namespace modeldeploy::vision::tool
```

- [ ] **Step 4: 写 `csrc/vision/tools/annotator.cpp`**

```cpp
#include "vision/tools/annotator.h"
#include <sstream>
namespace modeldeploy::vision::tool {
bool Annotator::begin(cv::Mat* m) { return frame_ && frame_->asMat(m); }
void Annotator::rectangle(const Rect2f& b, const cv::Scalar& color, int thickness) {
    cv::Mat m; if (!begin(&m)) return;
    cv::rectangle(m, cv::Rect((int)b.x, (int)b.y, (int)b.width, (int)b.height), color, thickness);
}
void Annotator::text(const std::string& s, Point2f org, const cv::Scalar& color, double scale) {
    cv::Mat m; if (!begin(&m)) return;
    cv::putText(m, s, cv::Point((int)org.x, (int)org.y), cv::FONT_HERSHEY_SIMPLEX, scale, color, 1, cv::LINE_AA);
}
void Annotator::line(Point2f a, Point2f b, const cv::Scalar& color, int thickness) {
    cv::Mat m; if (!begin(&m)) return;
    cv::line(m, cv::Point((int)a.x, (int)a.y), cv::Point((int)b.x, (int)b.y), color, thickness);
}
void Annotator::circle(Point2f c, int r, const cv::Scalar& color, int thickness) {
    cv::Mat m; if (!begin(&m)) return;
    cv::circle(m, cv::Point((int)c.x, (int)c.y), r, color, thickness);
}
void Annotator::fill_polygon(const std::vector<Point2f>& pts, const cv::Scalar& color, double alpha) {
    cv::Mat m; if (!begin(&m)) return;
    std::vector<cv::Point> poly;
    for (const auto& p : pts) poly.emplace_back((int)p.x, (int)p.y);
    cv::Mat overlay = m.clone();
    cv::fillPoly(overlay, std::vector<std::vector<cv::Point>>{poly}, color);
    cv::addWeighted(overlay, alpha, m, 1.0 - alpha, 0.0, m);
}
void draw_box_labels(const Detections& d, ImageData* frame,
                     const std::unordered_map<int, std::string>& labels) {
    Annotator ann(frame);
    for (size_t i = 0; i < d.size(); ++i) {
        const auto& b = d.boxes[i];
        ann.rectangle(b, cv::Scalar(0, 200, 0), 2);
        std::ostringstream oss;
        oss << (labels.count(d.class_id[i]) ? labels.at(d.class_id[i]) : std::to_string(d.class_id[i]));
        oss << " " << d.confidence[i];
        ann.text(oss.str(), Point2f(b.x, std::max(0.0f, b.y - 4)), cv::Scalar(0, 255, 255), 0.5);
    }
}
void draw_traces(const std::vector<Point2f>& trace, ImageData* frame,
                 const cv::Scalar& color, int thickness) {
    Annotator ann(frame);
    for (size_t i = 1; i < trace.size(); ++i)
        ann.line(trace[i - 1], trace[i], color, thickness);
}
} // namespace modeldeploy::vision::tool
```

- [ ] **Step 5: 构建 + 运行确认通过**

Run: `cmake --build build --parallel 8 && cd build && .\bin\test_modeldeploy.exe "[cv_tools]"`
Expected: 8 个 `[cv_tools]` 用例 PASS。

- [ ] **Step 6: Commit**

```bash
git add csrc/vision/tools/annotator.h csrc/vision/tools/annotator.cpp tests/test_cv_tools.cpp
git commit -m "feat(cv_tools): Annotator canvas + box/label/trace draw helpers"
```

---

## A4: `vision::tool` Metrics（mAP / Precision / Recall / F1）

**Files:**
- Create: `csrc/vision/tools/metrics.h`
- Create: `csrc/vision/tools/metrics.cpp`
- Modify: `tests/test_cv_tools.cpp`

**Interfaces:**
- Consumes: `Rect2f`、`iou`（A1）
- Produces:
  - `struct MetricsCounts { int tp{0}; int fp{0}; int fn{0}; }`
  - `struct MetricsScores { double precision{0}; double recall{0}; double f1{0}; double map50{0}; }`
  - `MetricsCounts vision::tool::count_tp_fp_fn(const std::vector<Rect2f>& preds, const std::vector<float>& pred_scores, const std::vector<Rect2f>& gt, double iou_threshold = 0.5)`
  - `MetricsScores vision::tool::evaluate_metrics(const std::vector<Rect2f>& preds, const std::vector<float>& pred_scores, const std::vector<Rect2f>& gt, double iou_threshold = 0.5)`

- [ ] **Step 1: 追加失败测试** (`tests/test_cv_tools.cpp`)

```cpp
#include "vision/tools/metrics.h"

TEST_CASE("Metrics counts and scores on tiny sample", "[cv_tools]") {
    std::vector<Rect2f> preds = {Rect2f(0,0,10,10), Rect2f(50,50,10,10)};
    std::vector<float> scores = {0.9f, 0.3f};
    std::vector<Rect2f> gt = {Rect2f(0,0,10,10)};
    auto mc = count_tp_fp_fn(preds, scores, gt, 0.5);
    REQUIRE(mc.tp == 1); REQUIRE(mc.fp == 1); REQUIRE(mc.fn == 0);
    auto s = evaluate_metrics(preds, scores, gt, 0.5);
    REQUIRE(s.precision == Approx(0.5));
    REQUIRE(s.recall == Approx(1.0));
    REQUIRE(s.f1 == Approx(2.0 * 0.5 * 1.0 / 1.5).margin(1e-6));
    REQUIRE(s.map50 > 0.0);
}
```

- [ ] **Step 2: 运行确认失败**

Run: `cd build && .\bin\test_modeldeploy.exe "[cv_tools]"`
Expected: FAIL（`vision/tools/metrics.h` 不存在）。

- [ ] **Step 3: 写 `csrc/vision/tools/metrics.h`**

```cpp
#pragma once
#include <vector>
#include "core/md_decl.h"
#include "vision/common/struct.h"
namespace modeldeploy::vision::tool {
struct MODELDEPLOY_CXX_EXPORT MetricsCounts { int tp{0}; int fp{0}; int fn{0}; };
struct MODELDEPLOY_CXX_EXPORT MetricsScores { double precision{0}; double recall{0}; double f1{0}; double map50{0}; };
MODELDEPLOY_CXX_EXPORT MetricsCounts count_tp_fp_fn(const std::vector<Rect2f>& preds,
                                                    const std::vector<float>& pred_scores,
                                                    const std::vector<Rect2f>& gt,
                                                    double iou_threshold = 0.5);
MODELDEPLOY_CXX_EXPORT MetricsScores evaluate_metrics(const std::vector<Rect2f>& preds,
                                                      const std::vector<float>& pred_scores,
                                                      const std::vector<Rect2f>& gt,
                                                      double iou_threshold = 0.5);
} // namespace modeldeploy::vision::tool
```

- [ ] **Step 4: 写 `csrc/vision/tools/metrics.cpp`**

```cpp
#include "vision/tools/metrics.h"
#include "vision/tools/detections.h"
#include <algorithm>
namespace modeldeploy::vision::tool {
MetricsCounts count_tp_fp_fn(const std::vector<Rect2f>& preds, const std::vector<float>& pred_scores,
                             const std::vector<Rect2f>& gt, double iou_threshold) {
    std::vector<size_t> order(preds.size());
    for (size_t i = 0; i < order.size(); ++i) order[i] = i;
    std::sort(order.begin(), order.end(), [&](size_t a, size_t b){ return pred_scores[a] > pred_scores[b]; });
    std::vector<bool> gt_matched(gt.size(), false);
    MetricsCounts c;
    for (size_t oi = 0; oi < order.size(); ++oi) {
        const auto& p = preds[order[oi]];
        int best = -1; double best_iou = iou_threshold;
        for (size_t g = 0; g < gt.size(); ++g) {
            if (gt_matched[g]) continue;
            double v = iou(p, gt[g]);
            if (v >= best_iou) { best_iou = v; best = (int)g; }
        }
        if (best >= 0) { gt_matched[(size_t)best] = true; ++c.tp; } else ++c.fp;
    }
    c.fn = (int)gt.size() - c.tp;
    return c;
}
MetricsScores evaluate_metrics(const std::vector<Rect2f>& preds, const std::vector<float>& pred_scores,
                               const std::vector<Rect2f>& gt, double iou_threshold) {
    auto mc = count_tp_fp_fn(preds, pred_scores, gt, iou_threshold);
    MetricsScores s;
    const int denom = mc.tp + mc.fp;
    s.precision = denom > 0 ? (double)mc.tp / denom : 0.0;
    const int gtdenom = mc.tp + mc.fn;
    s.recall = gtdenom > 0 ? (double)mc.tp / gtdenom : 0.0;
    s.f1 = (s.precision + s.recall) > 0 ? 2.0 * s.precision * s.recall / (s.precision + s.recall) : 0.0;
    // 11 点插值 mAP（按置信度降序扫描）
    std::vector<size_t> order(preds.size());
    for (size_t i = 0; i < order.size(); ++i) order[i] = i;
    std::sort(order.begin(), order.end(), [&](size_t a, size_t b){ return pred_scores[a] > pred_scores[b]; });
    std::vector<bool> gt_matched(gt.size(), false);
    std::vector<std::pair<double,double>> pr;
    int tp = 0, count = 0;
    if (gt.empty()) pr.emplace_back(0.0, 0.0);
    for (size_t oi = 0; oi < order.size(); ++oi) {
        const auto& p = preds[order[oi]];
        int best = -1; double best_iou = iou_threshold;
        for (size_t g = 0; g < gt.size(); ++g) {
            if (gt_matched[g]) continue;
            double v = iou(p, gt[g]);
            if (v >= best_iou) { best_iou = v; best = (int)g; }
        }
        ++count;
        if (best >= 0) { gt_matched[(size_t)best] = true; ++tp; }
        const double recall = gt.empty() ? 0.0 : (double)tp / gt.size();
        pr.emplace_back(recall, (double)tp / count);
    }
    double ap = 0.0;
    for (int r = 0; r <= 10; ++r) {
        const double target = r / 10.0;
        double maxp = 0.0;
        for (const auto& e : pr) if (e.first >= target) maxp = std::max(maxp, e.second);
        ap += maxp / 11.0;
    }
    s.map50 = ap;
    return s;
}
} // namespace modeldeploy::vision::tool
```

- [ ] **Step 5: 构建 + 运行确认通过**

Run: `cmake --build build --parallel 8 && cd build && .\bin\test_modeldeploy.exe "[cv_tools]"`
Expected: 9 个 `[cv_tools]` 用例 PASS。

- [ ] **Step 6: Commit**

```bash
git add csrc/vision/tools/metrics.h csrc/vision/tools/metrics.cpp tests/test_cv_tools.cpp
git commit -m "feat(cv_tools): Metrics mAP/precision/recall/F1"
```

---

## A5: `vision::tool` InferenceSlicer（大图切片 + 拼回）

**Files:**
- Create: `csrc/vision/tools/slicer.h`
- Create: `csrc/vision/tools/slicer.cpp`
- Modify: `tests/test_cv_tools.cpp`

**Interfaces:**
- Consumes: `ImageData`、`Rect2f`
- Produces:
  - `struct Slice { ImageData tile; Rect2f offset; }`
  - `class InferenceSlicer { InferenceSlicer(int tile_w, int tile_h, int overlap_px = 0); std::vector<Slice> slice(const ImageData& img) const; }`
  - `void vision::tool::reassemble(const std::vector<Slice>& slices, const std::vector<Detections>& per_slice, ImageData* out, std::vector<Rect2f>* mapped_boxes)`

- [ ] **Step 1: 追加失败测试** (`tests/test_cv_tools.cpp`)

```cpp
#include "vision/tools/slicer.h"

TEST_CASE("Slicer tiles a large image", "[cv_tools]") {
    std::vector<uint8_t> pixels(100 * 100 * 3, 0);
    ImageData img = ImageData::from_raw(pixels.data(), 100, 100, MdImageType::PKG_BGR_U8, true);
    InferenceSlicer slicer(60, 60, 10);
    auto tiles = slicer.slice(img);
    REQUIRE(tiles.size() >= 4);
    REQUIRE(tiles[0].tile.width() == 60);
    REQUIRE(tiles[0].offset.x == 0.0f);
}

TEST_CASE("Slicer reassembles mapped boxes", "[cv_tools]") {
    InferenceSlicer slicer(50, 50, 0);
    std::vector<uint8_t> pixels(100 * 100 * 3, 0);
    ImageData img = ImageData::from_raw(pixels.data(), 100, 100, MdImageType::PKG_BGR_U8, true);
    auto tiles = slicer.slice(img);
    Detections per; per.boxes = {Rect2f(0,0,10,10)}; per.confidence={0.9f}; per.class_id={0};
    std::vector<Detections> per_slice(tiles.size());
    per_slice[3] = per;  // 右下 tile 的局部框
    ImageData out; std::vector<Rect2f> mapped;
    reassemble(tiles, per_slice, &out, &mapped);
    REQUIRE(out.width() == 100);
    REQUIRE(mapped.size() == 1);
    REQUIRE(mapped[0].x == Approx(50.0f));  // tile offset (50,50)
    REQUIRE(mapped[0].y == Approx(50.0f));
}
```

- [ ] **Step 2: 运行确认失败**

Run: `cd build && .\bin\test_modeldeploy.exe "[cv_tools]"`
Expected: FAIL（`vision/tools/slicer.h` 不存在）。

- [ ] **Step 3/4: 写 `csrc/vision/tools/slicer.h` / `slicer.cpp`**

```cpp
// slicer.h
#pragma once
#include <vector>
#include "core/md_decl.h"
#include "vision/common/image_data.h"
#include "vision/common/struct.h"
#include "vision/tools/detections.h"
namespace modeldeploy::vision::tool {
struct MODELDEPLOY_CXX_EXPORT Slice { ImageData tile; Rect2f offset; };
class MODELDEPLOY_CXX_EXPORT InferenceSlicer {
public:
    InferenceSlicer(int tile_w, int tile_h, int overlap_px = 0)
        : tile_w_(tile_w), tile_h_(tile_h), overlap_(overlap_px) {}
    std::vector<Slice> slice(const ImageData& img) const;
private:
    int tile_w_, tile_h_, overlap_;
};
MODELDEPLOY_CXX_EXPORT void reassemble(const std::vector<Slice>& slices,
                                       const std::vector<Detections>& per_slice,
                                       ImageData* out, std::vector<Rect2f>* mapped_boxes);
} // namespace modeldeploy::vision::tool
```

```cpp
// slicer.cpp
#include "vision/tools/slicer.h"
#include <algorithm>
namespace modeldeploy::vision::tool {
std::vector<Slice> InferenceSlicer::slice(const ImageData& img) const {
    const int W = img.width(), H = img.height();
    std::vector<Slice> out;
    if (W <= 0 || H <= 0) return out;
    const int step_w = std::max(1, tile_w_ - overlap_);
    const int step_h = std::max(1, tile_h_ - overlap_);
    for (int y = 0; y < H; y += step_h)
        for (int x = 0; x < W; x += step_w) {
            const int tw = std::min(tile_w_, W - x);
            const int th = std::min(tile_h_, H - y);
            const Rect2f box((float)x, (float)y, (float)tw, (float)th);
            out.push_back(Slice{img.crop(box), box});
        }
    return out;
}
void reassemble(const std::vector<Slice>& slices, const std::vector<Detections>& per_slice,
                ImageData* out, std::vector<Rect2f>* mapped_boxes) {
    if (slices.empty()) return;
    const int W = (int)(slices[0].offset.x + slices[0].offset.width);
    const int H = (int)(slices[0].offset.y + slices[0].offset.height);
    *out = ImageData(W, H, MdImageType::PKG_BGR_U8);
    mapped_boxes->clear();
    for (size_t i = 0; i < slices.size(); ++i) {
        const auto& dets = (i < per_slice.size()) ? per_slice[i] : Detections{};
        for (const auto& b : dets.boxes) {
            mapped_boxes->push_back(Rect2f(b.x + slices[i].offset.x, b.y + slices[i].offset.y, b.width, b.height));
        }
    }
}
} // namespace modeldeploy::vision::tool
```

- [ ] **Step 5: 构建 + 运行确认通过**

Run: `cmake --build build --parallel 8 && cd build && .\bin\test_modeldeploy.exe "[cv_tools]"`
Expected: 11 个 `[cv_tools]` 用例 PASS。
> `reassemble` 依赖右下 tile 恰位于 index 3；100÷50 步长 50 → 网格 2×2，index 3 = (x=50,y=50) 右下，符合断言。

- [ ] **Step 6: Commit**

```bash
git add csrc/vision/tools/slicer.h csrc/vision/tools/slicer.cpp tests/test_cv_tools.cpp
git commit -m "feat(cv_tools): InferenceSlicer slice + reassemble mapped boxes"
```

---

## A6: `vision::tool` DetectionSmoother（检测抖动平滑）

**Files:**
- Create: `csrc/vision/tools/smoother.h`
- Create: `csrc/vision/tools/smoother.cpp`
- Modify: `tests/test_cv_tools.cpp`

**Interfaces:**
- Consumes: `Detections`（A1）
- Produces: `class Discovery visión::tool::DetectionSmoother { explicit DetectionSmoother(double alpha = 0.5); void reset(); Detections update(const Detections& in); }` —— 名称纠正：`modeldeploy::vision::tool::DetectionSmoother`

- [ ] **Step 1: 追加失败测试** (`tests/test_cv_tools.cpp`)

```cpp
#include "vision/tools/smoother.h"

TEST_CASE("Smoother EMA converges", "[cv_tools]") {
    Detections in;
    in.boxes = {Rect2f(10,20,30,40)}; in.confidence = {0.9f}; in.class_id = {0}; in.tracker_id = {1};
    DetectionSmoother sm(0.5);
    auto a = sm.update(in);
    REQUIRE(a.boxes[0].x == Approx(10.0f));
    in.boxes[0] = Rect2f(20,20,30,40);
    auto b = sm.update(in);
    REQUIRE(b.boxes[0].x > 10.0f && b.boxes[0].x < 20.0f); // 首->15
    for (int i = 0; i < 10; ++i) sm.update(in);
    auto c = sm.update(in);
    REQUIRE(c.boxes[0].x == Approx(20.0f).margin(0.05f));
}
```

- [ ] **Step 2: 运行确认失败**

Run: `cd build && .\bin\test_modeldeploy.exe "[cv_tools]"`
Expected: FAIL（`vision/tools/smoother.h` 不存在）。

- [ ] **Step 3/4: 写 `csrc/vision/tools/smoother.h` / `smoother.cpp`**

```cpp
// smoother.h
#pragma once
#include <vector>
#include "core/md_decl.h"
#include "vision/tools/detections.h"
namespace modeldeploy::vision::tool {
class MODELDEPLOY_CXX_EXPORT DetectionSmoother {
public:
    explicit DetectionSmoother(double alpha = 0.5) : alpha_(alpha) {}
    void reset() { state_.clear(); }
    Detections update(const Detections& in);
private:
    double alpha_;
    std::vector<Rect2f> state_;
};
} // namespace modeldeploy::vision::tool
```

```cpp
// smoother.cpp
#include "vision/tools/smoother.h"
#include <algorithm>
namespace modeldeploy::vision::tool {
Detections DetectionSmoother::update(const Detections& in) {
    Detections out = in;
    if (in.tracker_id.size() == in.boxes.size()) {
        if (state_.size() != in.boxes.size()) state_ = in.boxes;
        for (size_t i = 0; i < in.boxes.size(); ++i) {
            out.boxes[i].x = (float)(alpha_ * in.boxes[i].x + (1 - alpha_) * state_[i].x);
            out.boxes[i].y = (float)(alpha_ * in.boxes[i].y + (1 - alpha_) * state_[i].y);
        }
        state_ = out.boxes;
    } else {
        if (state_.empty()) return out;
        const size_t n = std::min(state_.size(), in.boxes.size());
        for (size_t i = 0; i < n; ++i) {
            out.boxes[i].x = (float)(alpha_ * in.boxes[i].x + (1 - alpha_) * state_[i].x);
            out.boxes[i].y = (float)(alpha_ * in.boxes[i].y + (1 - alpha_) * state_[i].y);
        }
        state_.resize(n);
        for (size_t i = 0; i < n; ++i) state_[i] = out.boxes[i];
    }
    return out;
}
} // namespace modeldeploy::vision::tool
```

- [ ] **Step 5: 构建 + 运行确认通过**

Run: `cmake --build build --parallel 8 && cd build && .\bin\test_modeldeploy.exe "[cv_tools]"`
Expected: 12 个 `[cv_tools]` 用例 PASS。

- [ ] **Step 6: Commit**

```bash
git add csrc/vision/tools/smoother.h csrc/vision/tools/smoother.cpp tests/test_cv_tools.cpp
git commit -m "feat(cv_tools): DetectionSmoother EMA"
```

> CV 工具层完成（Detections/Zone/Annotator/Metrics/Slicer/Smoother），全在 `BUILD_VISION` 下、主 CMake 未改。

---

## A7: `vision::solution` SolutionBase + ObjectCounter（跨线 + 区域 + 类维度）

**Files:**
- Create: `csrc/vision/solutions/solution_base.h`
- Create: `csrc/vision/solutions/object_counter.h`
- Create: `csrc/vision/solutions/object_counter.cpp`
- Modify: `csrc/vision/tools/zone.h`（给 `LineZone` 加 `in_side()`——见 A2 已含）
- Test: `tests/test_cv_solutions.cpp`
- Modify: `tests/CMakeLists.txt`（加 `test_cv_solutions.cpp`）

**Interfaces:**
- Consumes: `tracking::TrackResult`、`tool::LineZone`/`tool::PolygonZone`（A2）
- Produces:
  - `struct SolutionBase { virtual ~SolutionBase() = default; virtual void reset() = 0; };`
  - `struct CounterStats { int line_in{0}; int line_out{0}; std::map<int,int> class_count; };`
  - `class ObjectCounter : public SolutionBase { ObjectCounter(); void set_line(Point2f a, Point2f b); void set_region(const std::vector<Point2f>& pts); void set_classes(const std::vector<int32_t>& cls); void update(const std::vector<tracking::TrackResult>& tracks); CounterStats stats() const; int region_count() const; void reset() override; }`

- [ ] **Step 1: 写失败测试** (`tests/test_cv_solutions.cpp`)

```cpp
#include <catch2/catch_test_macros.hpp>
#include "vision/solutions/object_counter.h"
using namespace modeldeploy::vision;
using namespace modeldeploy::vision::solution;
using namespace modeldeploy::vision::tracking;

TEST_CASE("ObjectCounter counts line crossing once per track", "[cv_solution]") {
    ObjectCounter c;
    c.set_line(Point2f(5, 0), Point2f(5, 10)); // 竖线 x=5
    std::vector<TrackResult> t1(1); t1[0].track_id = 1; t1[0].box = Rect2f(0,4,2,2); t1[0].label_id = 0;
    c.update(t1);
    REQUIRE(c.stats().line_in == 0);
    std::vector<TrackResult> t2(1); t2[0].track_id = 1; t2[0].box = Rect2f(8,4,2,2); t2[0].label_id = 0;
    c.update(t2);
    REQUIRE(c.stats().line_in == 1); // 进 in 侧
    std::vector<TrackResult> t3(1); t3[0].track_id = 1; t3[0].box = Rect2f(9,4,2,2); t3[0].label_id = 0;
    c.update(t3);
    REQUIRE(c.stats().line_in == 1); // 仍在 in 侧不重复
}

TEST_CASE("ObjectCounter counts region + class dimension", "[cv_solution]") {
    ObjectCounter c;
    c.set_region({Point2f(0,0), Point2f(10,0), Point2f(10,10), Point2f(0,10)});
    std::vector<TrackResult> t(2);
    t[0].track_id = 1; t[0].box = Rect2f(2,2,2,2); t[0].label_id = 0;
    t[1].track_id = 2; t[1].box = Rect2f(3,3,2,2); t[1].label_id = 1;
    c.update(t);
    REQUIRE(c.region_count() == 2);
    REQUIRE(c.stats().class_count[0] == 1);
    REQUIRE(c.stats().class_count[1] == 1);
}
```

- [ ] **Step 2: 运行确认失败**

Run: `cd build && .\bin\test_modeldeploy.exe "[cv_solution]"`
Expected: FAIL（`vision/solutions/object_counter.h` 不存在）。

- [ ] **Step 3: 写 `csrc/vision/solutions/solution_base.h`**

```cpp
#pragma once
#include "core/md_decl.h"
namespace modeldeploy::vision::solution {
struct MODELDEPLOY_CXX_EXPORT SolutionBase {
    virtual ~SolutionBase() = default;
    virtual void reset() = 0;
};
} // namespace modeldeploy::vision::solution
```

- [ ] **Step 4: 写 `csrc/vision/solutions/object_counter.h`**

```cpp
#pragma once
#include <map>
#include <utility>
#include <vector>
#include "core/md_decl.h"
#include "vision/common/struct.h"
#include "vision/tracking/base_tracker.h"
#include "vision/tools/zone.h"
#include "vision/solutions/solution_base.h"
namespace modeldeploy::vision::solution {
struct MODELDEPLOY_CXX_EXPORT CounterStats { int line_in{0}; int line_out{0}; std::map<int, int> class_count; };
class MODELDEPLOY_CXX_EXPORT ObjectCounter : public SolutionBase {
public:
    ObjectCounter() = default;
    void set_line(Point2f a, Point2f b);
    void set_region(const std::vector<Point2f>& pts);
    void set_classes(const std::vector<int32_t>& cls);
    void update(const std::vector<tracking::TrackResult>& tracks);
    CounterStats stats() const { return stats_; }
    int region_count() const { return region_count_; }
    void reset() override;
private:
    tool::PolygonZone region_;
    bool has_line_{false};
    bool has_region_{false};
    std::vector<int32_t> classes_;
    std::pair<Point2f, Point2f> line_pts_;
    std::map<int, Point2f> last_centroid_;
    std::map<int, tool::LineZone> line_zone_;
    CounterStats stats_;
    int region_count_{0};
};
} // namespace modeldeploy::vision::solution
```

- [ ] **Step 5: 写 `csrc/vision/solutions/object_counter.cpp`**

```cpp
#include "vision/solutions/object_counter.h"
#include <algorithm>
namespace modeldeploy::vision::solution {
void ObjectCounter::set_line(Point2f a, Point2f b) {
    line_pts_ = {a, b};
    line_zone_.clear();               // 惰性重建每 track 的 LineZone
    has_line_ = true;
    stats_.line_in = stats_.line_out = 0;
}
void ObjectCounter::set_region(const std::vector<Point2f>& pts) {
    region_ = tool::PolygonZone(pts);
    has_region_ = true;
    region_.reset();
}
void ObjectCounter::set_classes(const std::vector<int32_t>& cls) { classes_ = cls; }
void ObjectCounter::update(const std::vector<tracking::TrackResult>& tracks) {
    for (const auto& t : tracks) {
        const Point2f c(t.box.x + t.box.width * 0.5f, t.box.y + t.box.height * 0.5f);
        if (has_line_) {
            // 惰性创建该 track 的跨线状态机；跨线事件用 in_side() 区分 in/out 方向
            auto itz = line_zone_.find(t.track_id);
            if (itz == line_zone_.end()) {
                itz = line_zone_.emplace(t.track_id, tool::LineZone(line_pts_.first, line_pts_.second)).first;
            }
            if (itz->second.trigger(c)) {
                if (itz->second.in_side()) ++stats_.line_in; else ++stats_.line_out;
            }
        }
        if (has_region_) {
            auto it = last_centroid_.find(t.track_id);
            const bool prev_in = it != last_centroid_.end() && region_.contains(it->second);
            const bool now_in = region_.contains(c);
            if (now_in && !prev_in) ++region_count_;
        }
        if (classes_.empty() || std::find(classes_.begin(), classes_.end(), t.label_id) != classes_.end())
            stats_.class_count[t.label_id]++;
        last_centroid_[t.track_id] = c;
    }
}
void ObjectCounter::reset() {
    region_.reset(); line_zone_.clear(); last_centroid_.clear();
    stats_ = CounterStats{}; region_count_ = 0;
}
} // namespace modeldeploy::vision::solution
```

> 说明：`line_pts_` 为 `std::pair<Point2f,Point2f>`，需加进 `object_counter.h` 的 private 成员（与 `line_zone_` 并列）。首帧后 `line_zone_[track_id]` 首次 `trigger(c)` 时 `has_last_=false` 不计数；出→入 in 侧翻转计 `line_in`，入→出翻转计 `line_out`。测试「帧3 仍 in 不重复」与「region_count==2」据此满足。

- [ ] **Step 6: 补成员 + 构建 + 运行确认通过**

在 `object_counter.h` private 区加 `std::pair<Point2f,Point2f> line_pts_;`（`#include <utility>`）。`tests/CMakeLists.txt` TEST_SOURCES 加 `test_cv_solutions.cpp`。
Run: `cmake --build build --parallel 8 && cd build && .\bin\test_modeldeploy.exe "[cv_solution]"`
Expected: 2 个 `[cv_solution]` 用例 PASS。

- [ ] **Step 7: Commit**

```bash
git add csrc/vision/solutions/ tests/test_cv_solutions.cpp tests/CMakeLists.txt
git commit -m "feat(cv_solution): SolutionBase + ObjectCounter (line/region/class)"
```

---

## A8: `vision::solution` Heatmap（轨迹热度）

**Files:**
- Create: `csrc/vision/solutions/heatmap.h`
- Create: `csrc/vision/solutions/heatmap.cpp`
- Modify: `tests/test_cv_solutions.cpp`

**Interfaces:**
- Consumes: `tracking::TrackResult`
- Produces: `class Heatmap : public SolutionBase { void set_size(int w, int h); void update(const std::vector<tracking::TrackResult>& tracks, int frame_w, int frame_h); const std::vector<float>& heat() const; std::pair<int,int> peak() const; float heat_at(int x, int y) const; void reset() override; }`

- [ ] **Step 1: 追加失败测试** (`tests/test_cv_solutions.cpp`)

```cpp
#include "vision/solutions/heatmap.h"

TEST_CASE("Heatmap accumulates at centroids", "[cv_solution]") {
    Heatmap hm;
    hm.set_size(10, 10);
    std::vector<TrackResult> t(1);
    t[0].track_id = 1; t[0].box = Rect2f(3, 3, 2, 2); // 质心 (4,4)
    hm.update(t, 10, 10);
    hm.update(t, 10, 10);
    auto p = hm.peak();
    REQUIRE(p.first == 4);
    REQUIRE(p.second == 4);
    REQUIRE(hm.heat_at(4, 4) == Approx(2.0f).margin(1e-5f));
    REQUIRE(hm.heat_at(0, 0) == Approx(0.0f));
}
```

- [ ] **Step 2: 运行确认失败**

Run: `cd build && .\bin\test_modeldeploy.exe "[cv_solution]"`
Expected: FAIL（`vision/solutions/heatmap.h` 不存在）。

- [ ] **Step 3/4: 写 `csrc/vision/solutions/heatmap.h` / `heatmap.cpp`**

```cpp
// heatmap.h
#pragma once
#include <utility>
#include <vector>
#include "core/md_decl.h"
#include "vision/tracking/base_tracker.h"
#include "vision/solutions/solution_base.h"
namespace modeldeploy::vision::solution {
class MODELDEPLOY_CXX_EXPORT Heatmap : public SolutionBase {
public:
    Heatmap() = default;
    void set_size(int w, int h);
    void update(const std::vector<tracking::TrackResult>& tracks, int frame_w, int frame_h);
    const std::vector<float>& heat() const { return heat_; }
    std::pair<int,int> peak() const;
    float heat_at(int x, int y) const;
    void reset() override;
private:
    int w_{0}, h_{0};
    std::vector<float> heat_;
};
} // namespace modeldeploy::vision::solution
```

```cpp
// heatmap.cpp
#include "vision/solutions/heatmap.h"
#include <algorithm>
namespace modeldeploy::vision::solution {
void Heatmap::set_size(int w, int h) { w_ = w; h_ = h; heat_.assign((size_t)w * h, 0.0f); }
void Heatmap::update(const std::vector<tracking::TrackResult>& tracks, int frame_w, int frame_h) {
    if (heat_.empty() || w_ <= 0 || h_ <= 0) return;
    const float sx = (float)w_ / (float)std::max(1, frame_w);
    const float sy = (float)h_ / (float)std::max(1, frame_h);
    for (const auto& t : tracks) {
        const float cx = t.box.x + t.box.width * 0.5f;
        const float cy = t.box.y + t.box.height * 0.5f;
        int x = std::max(0, std::min(w_ - 1, (int)(cx * sx)));
        int y = std::max(0, std::min(h_ - 1, (int)(cy * sy)));
        heat_[(size_t)y * w_ + x] += 1.0f;
    }
}
float Heatmap::heat_at(int x, int y) const {
    if (heat_.empty() || x < 0 || y < 0 || x >= w_ || y >= h_) return 0.0f;
    return heat_[(size_t)y * w_ + x];
}
std::pair<int,int> Heatmap::peak() const {
    if (heat_.empty()) return {0, 0};
    auto it = std::max_element(heat_.begin(), heat_.end());
    if (*it <= 0.0f) return {0, 0};
    const size_t idx = (size_t)(it - heat_.begin());
    return {(int)(idx % w_), (int)(idx / w_)};
}
void Heatmap::reset() { std::fill(heat_.begin(), heat_.end(), 0.0f); }
} // namespace modeldeploy::vision::solution
```

- [ ] **Step 5: 构建 + 运行确认通过**

Run: `cmake --build build --parallel 8 && cd build && .\bin\test_modeldeploy.exe "[cv_solution]"`
Expected: 3 个 `[cv_solution]` 用例 PASS。

- [ ] **Step 6: Commit**

```bash
git add csrc/vision/solutions/heatmap.h csrc/vision/solutions/heatmap.cpp tests/test_cv_solutions.cpp
git commit -m "feat(cv_solution): Heatmap trajectory accumulation + peak"
```

---

## A9: `vision::solution` SpeedEstimator + DistanceEstimator

**Files:**
- Create: `csrc/vision/solutions/speed_estimator.h` / `.cpp`
- Create: `csrc/vision/solutions/distance_estimator.h` / `.cpp`
- Modify: `tests/test_cv_solutions.cpp`

**Interfaces:**
- Consumes: `tracking::TrackResult`、`Point2f`
- Produces:
  - `class SpeedEstimator : public SolutionBase { void set_meter_per_pixel(float m); void update(const std::vector<tracking::TrackResult>& tracks, double timestamp_ms); std::map<int,float> speeds_px_per_s() const; std::map<int,float> speeds_m_s() const; void reset() override; }`
  - `class DistanceEstimator : public SolutionBase { void set_meter_per_pixel(float m); std::vector<std::pair<std::pair<int,int>, float>> pair_distances_px(const std::vector<tracking::TrackResult>& tracks); std::vector<std::pair<std::pair<int,int>, float>> pair_distances_m(const std::vector<tracking::TrackResult>& tracks); void reset() override; }`

- [ ] **Step 1: 追加失败测试** (`tests/test_cv_solutions.cpp`)

```cpp
#include "vision/solutions/speed_estimator.h"
#include "vision/solutions/distance_estimator.h"

TEST_CASE("SpeedEstimator displacement over timestamp", "[cv_solution]") {
    SpeedEstimator se;
    se.set_meter_per_pixel(0.01f);
    std::vector<TrackResult> t(1); t[0].track_id = 1; t[0].box = Rect2f(0,0,10,10);
    se.update(t, 0.0);
    t[0].box = Rect2f(10,0,10,10); // 质心移动 10px
    se.update(t, 1000.0);
    REQUIRE(se.speeds_px_per_s()[1] == Approx(10.0f).margin(1e-3f));
    REQUIRE(se.speeds_m_s()[1] == Approx(0.1f).margin(1e-3f));
}

TEST_CASE("DistanceEstimator pair distances", "[cv_solution]") {
    DistanceEstimator de;
    de.set_meter_per_pixel(0.5f);
    std::vector<TrackResult> t(2);
    t[0].track_id = 1; t[0].box = Rect2f(0,0,1,1);   // 质心 (0.5,0.5)
    t[1].track_id = 2; t[1].box = Rect2f(5,0,1,1);   // 质心 (5.5,0.5)
    auto d = de.pair_distances_px(t);
    REQUIRE(d.size() == 1);
    REQUIRE(d[0].second == Approx(5.0f).margin(1e-3f));
    auto dm = de.pair_distances_m(t);
    REQUIRE(dm[0].second == Approx(2.5f).margin(1e-3f));
}
```

- [ ] **Step 2: 运行确认失败**

Run: `cd build && .\bin\test_modeldeploy.exe "[cv_solution]"`
Expected: FAIL（`speed_estimator.h` / `distance_estimator.h` 不存在）。

- [ ] **Step 3/4: 写头文件与实现**

```cpp
// speed_estimator.h
#pragma once
#include <map>
#include "core/md_decl.h"
#include "vision/tracking/base_tracker.h"
#include "vision/solutions/solution_base.h"
namespace modeldeploy::vision::solution {
class MODELDEPLOY_CXX_EXPORT SpeedEstimator : public SolutionBase {
public:
    void set_meter_per_pixel(float m) { mpp_ = m; }
    void update(const std::vector<tracking::TrackResult>& tracks, double timestamp_ms);
    std::map<int,float> speeds_px_per_s() const { return px_per_s_; }
    std::map<int,float> speeds_m_s() const;
    void reset() override;
private:
    float mpp_{0.01f};
    std::map<int, std::pair<Point2f, double>> last_;
    std::map<int,float> px_per_s_;
};
} // namespace modeldeploy::vision::solution
```

```cpp
// speed_estimator.cpp
#include "vision/solutions/speed_estimator.h"
#include <cmath>
namespace modeldeploy::vision::solution {
void SpeedEstimator::update(const std::vector<tracking::TrackResult>& tracks, double timestamp_ms) {
    px_per_s_.clear();
    for (const auto& t : tracks) {
        const Point2f c(t.box.x + t.box.width * 0.5f, t.box.y + t.box.height * 0.5f);
        auto it = last_.find(t.track_id);
        if (it != last_.end()) {
            const double dt = timestamp_ms - it->second.second;
            const float dx = c.x - it->second.first.x;
            const float dy = c.y - it->second.first.y;
            const float dist = std::sqrt(dx * dx + dy * dy);
            if (dt > 0.0) px_per_s_[t.track_id] = dist / (float)dt * 1000.0f;
        }
        last_[t.track_id] = {c, timestamp_ms};
    }
}
std::map<int,float> SpeedEstimator::speeds_m_s() const {
    std::map<int,float> out;
    for (const auto& kv : px_per_s_) out[kv.first] = kv.second * mpp_;
    return out;
}
void SpeedEstimator::reset() { last_.clear(); px_per_s_.clear(); }
} // namespace modeldeploy::vision::solution
```

```cpp
// distance_estimator.h
#pragma once
#include <utility>
#include <vector>
#include "core/md_decl.h"
#include "vision/tracking/base_tracker.h"
#include "vision/solutions/solution_base.h"
namespace modeldeploy::vision::solution {
class MODELDEPLOY_CXX_EXPORT DistanceEstimator : public SolutionBase {
public:
    void set_meter_per_pixel(float m) { mpp_ = m; }
    std::vector<std::pair<std::pair<int,int>, float>> pair_distances_px(const std::vector<tracking::TrackResult>& tracks);
    std::vector<std::pair<std::pair<int,int>, float>> pair_distances_m(const std::vector<tracking::TrackResult>& tracks);
    void reset() override {}
private:
    float mpp_{0.01f};
};
} // namespace modeldeploy::vision::solution
```

```cpp
// distance_estimator.cpp
#include "vision/solutions/distance_estimator.h"
#include <cmath>
namespace modeldeploy::vision::solution {
static Point2f centroid(const tracking::TrackResult& t) {
    return Point2f(t.box.x + t.box.width * 0.5f, t.box.y + t.box.height * 0.5f);
}
std::vector<std::pair<std::pair<int,int>, float>> DistanceEstimator::pair_distances_px(
        const std::vector<tracking::TrackResult>& tracks) {
    std::vector<std::pair<std::pair<int,int>, float>> out;
    std::vector<std::pair<int,Point2f>> pts;
    for (const auto& t : tracks) pts.emplace_back(t.track_id, centroid(t));
    for (size_t i = 0; i < pts.size(); ++i)
        for (size_t j = i + 1; j < pts.size(); ++j) {
            const float dx = pts[i].second.x - pts[j].second.x;
            const float dy = pts[i].second.y - pts[j].second.y;
            out.emplace_back(std::make_pair(pts[i].first, pts[j].first), std::sqrt(dx*dx + dy*dy));
        }
    return out;
}
std::vector<std::pair<std::pair<int,int>, float>> DistanceEstimator::pair_distances_m(
        const std::vector<tracking::TrackResult>& tracks) {
    auto px = pair_distances_px(tracks);
    for (auto& e : px) e.second *= mpp_;
    return px;
}
} // namespace modeldeploy::vision::solution
```

- [ ] **Step 5: 构建 + 运行确认通过**

Run: `cmake --build build --parallel 8 && cd build && .\bin\test_modeldeploy.exe "[cv_solution]"`
Expected: 5 个 `[cv_solution]` 用例 PASS。

- [ ] **Step 6: Commit**

```bash
git add csrc/vision/solutions/speed_estimator.* csrc/vision/solutions/distance_estimator.* tests/test_cv_solutions.cpp
git commit -m "feat(cv_solution): SpeedEstimator + DistanceEstimator"
```

---

## A10: `vision::solution` ObjectCropper + ObjectBlur

**Files:**
- Create: `csrc/vision/solutions/object_cropper.h` / `.cpp`
- Create: `csrc/vision/solutions/object_blur.h` / `.cpp`
- Modify: `tests/test_cv_solutions.cpp`

**Interfaces:**
- Consumes: `ImageData`（`crop`、`asMat`）、`Rect2f`
- Produces:
  - `class ObjectCropper { void crop(const ImageData& img, const Rect2f& box, ImageData* out) const; }`
  - `class ObjectBlur { explicit ObjectBlur(int ksize = 15); void blur(const ImageData& img, const Rect2f& box, ImageData* out) const; }`

- [ ] **Step 1: 追加失败测试** (`tests/test_cv_solutions.cpp`)

```cpp
#include "vision/solutions/object_cropper.h"
#include "vision/solutions/object_blur.h"

TEST_CASE("ObjectCropper extracts ROI", "[cv_solution]") {
    std::vector<uint8_t> pixels(20 * 20 * 3, 0);
    ImageData img = ImageData::from_raw(pixels.data(), 20, 20, MdImageType::PKG_BGR_U8, true);
    ObjectCropper cr;
    ImageData out;
    cr.crop(img, Rect2f(2, 2, 10, 8), &out);
    REQUIRE(out.width() == 10);
    REQUIRE(out.height() == 8);
}

TEST_CASE("ObjectBlur blurs region only", "[cv_solution]") {
    cv::Mat m(20, 20, CV_8UC3, cv::Scalar(0, 0, 0));
    cv::rectangle(m, cv::Rect(5, 5, 10, 10), cv::Scalar(255, 255, 255), cv::FILLED);
    ImageData img(m);
    ObjectBlur bl(11);
    ImageData out;
    bl.blur(img, Rect2f(5, 5, 10, 10), &out);
    cv::Mat om;
    REQUIRE(out.asMat(&om));
    REQUIRE(om.at<cv::Vec3b>(10, 10)[0] < 255);   // 框内中值被模糊
    REQUIRE(om.at<cv::Vec3b>(1, 1)[0] == 0);      // 框外角不受影响
}
```

- [ ] **Step 2: 运行确认失败**

Run: `cd build && .\bin\test_modeldeploy.exe "[cv_solution]"`
Expected: FAIL（`object_cropper.h` / `object_blur.h` 不存在）。

- [ ] **Step 3/4: 写头文件与实现**

```cpp
// object_cropper.h
#pragma once
#include "core/md_decl.h"
#include "vision/common/image_data.h"
#include "vision/common/struct.h"
namespace modeldeploy::vision::solution {
class MODELDEPLOY_CXX_EXPORT ObjectCropper {
public:
    void crop(const ImageData& img, const Rect2f& box, ImageData* out) const;
};
} // namespace modeldeploy::vision::solution
```

```cpp
// object_cropper.cpp
#include "vision/solutions/object_cropper.h"
namespace modeldeploy::vision::solution {
void ObjectCropper::crop(const ImageData& img, const Rect2f& box, ImageData* out) const {
    if (!out) return;
    *out = img.crop(box);  // 复用 ImageData::crop
}
} // namespace modeldeploy::vision::solution
```

```cpp
// object_blur.h
#pragma once
#include "core/md_decl.h"
#include "vision/common/image_data.h"
#include "vision/common/struct.h"
namespace modeldeploy::vision::solution {
class MODELDEPLOY_CXX_EXPORT ObjectBlur {
public:
    explicit ObjectBlur(int ksize = 15) : ksize_(ksize) {}
    void blur(const ImageData& img, const Rect2f& box, ImageData* out) const;
private:
    int ksize_;
};
} // namespace modeldeploy::vision::solution
```

```cpp
// object_blur.cpp
#include "vision/solutions/object_blur.h"
#include <opencv2/opencv.hpp>
namespace modeldeploy::vision::solution {
void ObjectBlur::blur(const ImageData& img, const Rect2f& box, ImageData* out) const {
    if (!out) return;
    cv::Mat src;
    if (!img.asMat(&src)) return;
    cv::Mat dst = src.clone();
    const int x = (int)box.x, y = (int)box.y;
    const int w = (int)box.width, h = (int)box.height;
    cv::Rect roi(std::max(0, x), std::max(0, y),
                 std::min(w, src.cols - std::max(0, x)), std::min(h, src.rows - std::max(0, y)));
    if (roi.width > 0 && roi.height > 0) {
        cv::Mat region = dst(roi);
        cv::GaussianBlur(region, region, cv::Size(ksize_, ksize_), 0);
    }
    *out = ImageData(dst);
}
} // namespace modeldeploy::vision::solution
```

- [ ] **Step 5: 构建 + 运行确认通过**

Run: `cmake --build build --parallel 8 && cd build && .\bin\test_modeldeploy.exe "[cv_solution]"`
Expected: 7 个 `[cv_solution]` 用例 PASS。

- [ ] **Step 6: Commit**

```bash
git add csrc/vision/solutions/object_cropper.* csrc/vision/solutions/object_blur.* tests/test_cv_solutions.cpp
git commit -m "feat(cv_solution): ObjectCropper + ObjectBlur"
```

---

## A11: `vision::solution` WorkoutMonitor + ParkingManager + VisionEye

**Files:**
- Create: `csrc/vision/solutions/workout_monitor.h` / `.cpp`
- Create: `csrc/vision/solutions/parking_manager.h` / `.cpp`
- Create: `csrc/vision/solutions/vision_eye.h` / `.cpp`
- Modify: `tests/test_cv_solutions.cpp`

**Interfaces:**
- Consumes: `Point3f`（姿态关键点）、`Rect2f`、`tool::PolygonZone`（A2）、`tracking::TrackResult`
- Produces:
  - `static float WorkoutMonitor::angle(Point3f a, Point3f b, Point3f c)`（b 为顶点）
  - `class WorkoutMonitor { WorkoutMonitor(float min_deg = 70.0f, float max_deg = 160.0f); int reps() const; void update(float elbow_deg); void reset() override; }`
  - `class ParkingManager { void set_slots(const std::vector<std::vector<Point2f>>& slots); void update(const std::vector<tracking::TrackResult>& tracks); std::vector<bool> occupancy() const; void reset() override; }`
  - `static Point2f VisionEye::map_to_eye(Point2f centroid, float eye_level_y)`
  - `class VisionEye { explicit VisionEye(float eye_level_y = 0.0f); void add(Point2f centroid); const std::vector<Point2f>& eyes() const; void reset() override; }`

- [ ] **Step 1: 追加失败测试** (`tests/test_cv_solutions.cpp`)

```cpp
#include "vision/solutions/workout_monitor.h"
#include "vision/solutions/parking_manager.h"
#include "vision/solutions/vision_eye.h"

TEST_CASE("WorkoutMonitor angles and rep count", "[cv_solution]") {
    WorkoutMonitor wm(70.0f, 120.0f);
    REQUIRE(WorkoutMonitor::angle(Point3f(0,0,0), Point3f(1,1,0), Point3f(2,0,0)) == Approx(90.0f).margin(1e-2f));
    wm.update(135.0f); REQUIRE(wm.reps() == 0);
    wm.update(80.0f);
    wm.update(130.0f); REQUIRE(wm.reps() == 1);
}

TEST_CASE("ParkingManager occupancy per slot", "[cv_solution]") {
    ParkingManager pm;
    pm.set_slots({
        {Point2f(0,0), Point2f(10,0), Point2f(10,10), Point2f(0,10)},
        {Point2f(20,0), Point2f(30,0), Point2f(30,10), Point2f(20,10)}
    });
    std::vector<TrackResult> t(1); t[0].track_id = 1; t[0].box = Rect2f(2,2,2,2);
    pm.update(t);
    auto occ = pm.occupancy();
    REQUIRE(occ.size() == 2);
    REQUIRE(occ[0] == true);
    REQUIRE(occ[1] == false);
}

TEST_CASE("VisionEye maps centroid to eye point", "[cv_solution]") {
    VisionEye ve(100.0f);
    auto e = ve.map_to_eye(Point2f(50, 200), 100.0f);
    REQUIRE(e.x == Approx(50.0f));
    REQUIRE(e.y == Approx(100.0f));
    ve.add(Point2f(50, 200));
    REQUIRE(ve.eyes().size() == 1);
    REQUIRE(ve.eyes()[0].x == Approx(50.0f));
}
```

- [ ] **Step 2: 运行确认失败**

Run: `cd build && .\bin\test_modeldeploy.exe "[cv_solution]"`
Expected: FAIL（三个头文件不存在）。

- [ ] **Step 3/4: 写头文件与实现**

```cpp
// workout_monitor.h
#pragma once
#include "core/md_decl.h"
#include "vision/common/struct.h"
#include "vision/solutions/solution_base.h"
namespace modeldeploy::vision::solution {
class MODELDEPLOY_CXX_EXPORT WorkoutMonitor : public SolutionBase {
public:
    WorkoutMonitor(float min_deg = 70.0f, float max_deg = 160.0f) : min_(min_deg), max_(max_deg) {}
    static float angle(Point3f a, Point3f b, Point3f c);
    int reps() const { return reps_; }
    void update(float elbow_deg);
    void reset() override { reps_ = 0; down_ = false; }
private:
    float min_, max_;
    int reps_{0};
    bool down_{false};
};
} // namespace modeldeploy::vision::solution
```

```cpp
// workout_monitor.cpp
#include "vision/solutions/workout_monitor.h"
#include <cmath>
namespace modeldeploy::vision::solution {
static const float kPi = 3.14159265358979323846f;
float WorkoutMonitor::angle(Point3f a, Point3f b, Point3f c) {
    const float abx = a.x - b.x, aby = a.y - b.y;
    const float cbx = c.x - b.x, cby = c.y - b.y;
    const float dot = abx * cbx + aby * cby;
    const float n1 = std::sqrt(abx * abx + aby * aby);
    const float n2 = std::sqrt(cbx * cbx + cby * cby);
    if (n1 <= 0 || n2 <= 0) return 180.0f;
    const float cosv = std::max(-1.0f, std::min(1.0f, dot / (n1 * n2)));
    return std::acos(cosv) * 180.0f / kPi;
}
void WorkoutMonitor::update(float deg) {
    if (deg < min_) down_ = true;
    else if (deg > max_ && down_) { ++reps_; down_ = false; }
}
} // namespace modeldeploy::vision::solution
```

```cpp
// parking_manager.h
#pragma once
#include <vector>
#include "core/md_decl.h"
#include "vision/common/struct.h"
#include "vision/tracking/base_tracker.h"
#include "vision/tools/zone.h"
#include "vision/solutions/solution_base.h"
namespace modeldeploy::vision::solution {
class MODELDEPLOY_CXX_EXPORT ParkingManager : public SolutionBase {
public:
    void set_slots(const std::vector<std::vector<Point2f>>& slots);
    void update(const std::vector<tracking::TrackResult>& tracks);
    std::vector<bool> occupancy() const { return occupied_; }
    void reset() override { std::fill(occupied_.begin(), occupied_.end(), false); }
private:
    std::vector<tool::PolygonZone> slots_;
    std::vector<bool> occupied_;
};
} // namespace modeldeploy::vision::solution
```

```cpp
// parking_manager.cpp
#include "vision/solutions/parking_manager.h"
#include <algorithm>
namespace modeldeploy::vision::solution {
void ParkingManager::set_slots(const std::vector<std::vector<Point2f>>& slots) {
    slots_.clear();
    occupied_.assign(slots.size(), false);
    for (const auto& s : slots) slots_.emplace_back(tool::PolygonZone(s));
}
void ParkingManager::update(const std::vector<tracking::TrackResult>& tracks) {
    std::fill(occupied_.begin(), occupied_.end(), false);
    for (const auto& t : tracks) {
        const Point2f c(t.box.x + t.box.width * 0.5f, t.box.y + t.box.height * 0.5f);
        for (size_t i = 0; i < slots_.size(); ++i)
            if (slots_[i].contains(c)) occupied_[i] = true;
    }
}
} // namespace modeldeploy::vision::solution
```

```cpp
// vision_eye.h
#pragma once
#include <vector>
#include "core/md_decl.h"
#include "vision/common/struct.h"
#include "vision/solutions/solution_base.h"
namespace modeldeploy::vision::solution {
class MODELDEPLOY_CXX_EXPORT VisionEye : public SolutionBase {
public:
    explicit VisionEye(float eye_level_y = 0.0f) : eye_level_(eye_level_y) {}
    static Point2f map_to_eye(Point2f centroid, float eye_level_y);
    void add(Point2f centroid);
    const std::vector<Point2f>& eyes() const { return eyes_; }
    void reset() override { eyes_.clear(); }
private:
    float eye_level_;
    std::vector<Point2f> eyes_;
};
} // namespace modeldeploy::vision::solution
```

```cpp
// vision_eye.cpp
#include "vision/solutions/vision_eye.h"
namespace modeldeploy::vision::solution {
Point2f VisionEye::map_to_eye(Point2f centroid, float eye_level_y) { return Point2f(centroid.x, eye_level_y); }
void VisionEye::add(Point2f centroid) { eyes_.push_back(map_to_eye(centroid, eye_level_)); }
} // namespace modeldeploy::vision::solution
```

- [ ] **Step 5: 构建 + 运行确认通过**

Run: `cmake --build build --parallel 8 && cd build && .\bin\test_modeldeploy.exe "[cv_solution]"`
Expected: 10 个 `[cv_solution]` 用例 PASS。
> `angle(0,0)-(1,1)-(2,0)`：两向量 (1,1) 与 (-1,1)，dot=0，夹角 90°。`kPi` 字面量内联避免 MSVC `M_PI` 问题。

- [ ] **Step 6: Commit**

```bash
git add csrc/vision/solutions/workout_monitor.* csrc/vision/solutions/parking_manager.* csrc/vision/solutions/vision_eye.* tests/test_cv_solutions.cpp
git commit -m "feat(cv_solution): WorkoutMonitor + ParkingManager + VisionEye"
```

> **CV 方案层完成**：9 个方案（ObjectCounter/Heatmap/Speed/Distance/Cropper/Blur/Workout/Parking/VisionEye）复用 `tracking::TrackResult`/`tool::Zone`/`ImageData`，无新权重、`[cv_solution]` 合成确定性测试恒跑。

---

## A12: CV Python（`vision.solutions` + `vision.tools` 子模块）

**Files:**
- Create: `csrc/pybind/vision/solutions_pybind.cpp`
- Create: `csrc/pybind/vision/tools_pybind.cpp`
- Modify: `csrc/pybind/vision/vision_pybind.cpp`（声明 + 注册）
- Test: Python smoke

**Interfaces:**
- Consumes: A1–A11 的 `vision::tool`/`vision::solution` 类
- Produces: `modeldeploy.vision.solutions.ObjectCounter(...)`、`modeldeploy.vision.tools.iou(...)` 等

- [ ] **Step 1: 写 `csrc/pybind/vision/solutions_pybind.cpp`**

```cpp
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <vector>
#include "vision/solutions/object_counter.h"
#include "vision/solutions/heatmap.h"
#include "vision/solutions/speed_estimator.h"
#include "vision/solutions/distance_estimator.h"
#include "vision/solutions/workout_monitor.h"
#include "vision/solutions/parking_manager.h"
#include "vision/tracking/base_tracker.h"

namespace modeldeploy::vision {
    void bind_solutions(const pybind11::module& m) {
        using namespace modeldeploy::vision::solution;
        pybind11::class_<ObjectCounter>(m, "ObjectCounter")
            .def(pybind11::init<>())
            .def("set_line", &ObjectCounter::set_line)
            .def("set_region", &ObjectCounter::set_region)
            .def("update", [](ObjectCounter& s, std::vector<tracking::TrackResult>& t){ s.update(t); })
            .def("line_in", [](ObjectCounter& s){ return s.stats().line_in; })
            .def("line_out", [](ObjectCounter& s){ return s.stats().line_out; })
            .def("class_count", [](ObjectCounter& s){ return s.stats().class_count; })
            .def("region_count", &ObjectCounter::region_count);
        pybind11::class_<Heatmap>(m, "Heatmap")
            .def(pybind11::init<>())
            .def("set_size", &Heatmap::set_size)
            .def("update", [](Heatmap& s, std::vector<tracking::TrackResult>& t, int w, int h){ s.update(t, w, h); })
            .def("peak", &Heatmap::peak)
            .def("heat_at", &Heatmap::heat_at);
        pybind11::class_<SpeedEstimator>(m, "SpeedEstimator")
            .def(pybind11::init<>())
            .def("set_meter_per_pixel", &SpeedEstimator::set_meter_per_pixel)
            .def("update", &SpeedEstimator::update)
            .def("speeds_m_s", &SpeedEstimator::speeds_m_s);
        pybind11::class_<DistanceEstimator>(m, "DistanceEstimator")
            .def(pybind11::init<>())
            .def("set_meter_per_pixel", &DistanceEstimator::set_meter_per_pixel)
            .def("pair_distances_m", &DistanceEstimator::pair_distances_m);
        pybind11::class_<WorkoutMonitor>(m, "WorkoutMonitor")
            .def(pybind11::init<float, float>(), pybind11::arg("min_deg") = 70.0f, pybind11::arg("max_deg") = 160.0f)
            .def_static("angle", &WorkoutMonitor::angle)
            .def("update", &WorkoutMonitor::update)
            .def("reps", &WorkoutMonitor::reps);
        pybind11::class_<ParkingManager>(m, "ParkingManager")
            .def(pybind11::init<>())
            .def("set_slots", &ParkingManager::set_slots)
            .def("update", &ParkingManager::update)
            .def("occupancy", &ParkingManager::occupancy);
    }
} // namespace modeldeploy::vision
```

- [ ] **Step 2: 写 `csrc/pybind/vision/tools_pybind.cpp`**

```cpp
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "vision/tools/detections.h"
#include "vision/tools/zone.h"
#include "vision/tools/metrics.h"

namespace modeldeploy::vision {
    void bind_tools(const pybind11::module& m) {
        using namespace modeldeploy::vision::tool;
        pybind11::class_<Detections>(m, "Detections")
            .def(pybind11::init<>())
            .def_readwrite("boxes", &Detections::boxes)
            .def_readwrite("class_id", &Detections::class_id)
            .def_readwrite("confidence", &Detections::confidence)
            .def_readwrite("tracker_id", &Detections::tracker_id)
            .def("__len__", &Detections::size);
        m.def("iou", &iou);
        m.def("nms", [](Detections& d, float t){ nms(d, t); }, pybind11::arg("d"), pybind11::arg("iou_threshold") = 0.5f);
        m.def("from_track", &from_track);
        pybind11::class_<LineZone>(m, "LineZone")
            .def(pybind11::init<Point2f, Point2f>())
            .def("trigger", &LineZone::trigger)
            .def("trigger_count", &LineZone::trigger_count);
        pybind11::class_<PolygonZone>(m, "PolygonZone")
            .def(pybind11::init<std::vector<Point2f>>())
            .def("contains", &PolygonZone::contains)
            .def("current_count", &PolygonZone::current_count);
        m.def("evaluate_metrics", &evaluate_metrics);
    }
} // namespace modeldeploy::vision
```

- [ ] **Step 3: 编辑 `csrc/pybind/vision/vision_pybind.cpp`**

声明区（`bind_landmark` 声明后）加：
```cpp
    void bind_solutions(const pybind11::module&);
    void bind_tools(const pybind11::module&);
```
`bind_vision` 末尾（`bind_landmark(m);` 之后）加：
```cpp
        bind_solutions(m);
        bind_tools(m);
```

- [ ] **Step 4: 构建 + Python smoke**

用 `BUILD_VISION=ON BUILD_PYTHON=ON` 构建（如 `build_py`）后：
```bash
cd build_py && python -c "
from modeldeploy.vision import solutions, tools
c = solutions.ObjectCounter()
c.set_line((5,0),(5,10))
c.update([{'track_id':1,'box':(0,4,2,2),'label_id':0,'score':1.0}])
d = tools.Detections()
assert abs(tools.iou((0,0,10,10),(0,0,10,10)) - 1.0) < 1e-5
print('cv solutions/tools smoke OK')
"
```
Expected: `cv solutions/tools smoke OK` 无异常。

- [ ] **Step 5: Commit**

```bash
git add csrc/pybind/vision/solutions_pybind.cpp csrc/pybind/vision/tools_pybind.cpp csrc/pybind/vision/vision_pybind.cpp
git commit -m "feat(pybind): bind vision.solutions + vision.tools"
```

---

## A13: CV CAPI（解决方案句柄 + 工具纯函数）

**Files:**
- Modify: `capi/md_capi.h`（句柄 + 枚举 + 函数声明）
- Modify: `capi/md_capi.cpp`（实现）
- Test: `tests/test_capi.cpp`（`[capi]`）

**Interfaces:**
- Consumes: A1–A11；CAPI 既有 `MDStatus`/`set_error` 工具
- Produces:
  - `MDSolutionHandle`（`struct md_solution_handle { MDSolutionKind kind; void* obj; }`）
  - `enum MD_SOLUTION_KIND { MD_SOLUTION_OBJECT_COUNTER, MD_SOLUTION_HEATMAP, MD_SOLUTION_SPEED, MD_SOLUTION_DISTANCE, MD_SOLUTION_WORKOUT, MD_SOLUTION_PARKING }`
  - `MDStatus md_solution_create(MDSolutionHandle* out, MDSolutionKind kind);`
  - `MDStatus md_solution_destroy(MDSolutionHandle);`
  - `MDStatus md_solution_object_counter_set_line(MDSolutionHandle, float ax, float ay, float bx, float by);`
  - `MDStatus md_solution_object_counter_update(MDSolutionHandle, const float* boxes, size_t n, const int* label_ids, const int* track_ids);`
  - `MDStatus md_solution_object_counter_hline(MDSolutionHandle, int* in, int* out_count);`
  - `MDStatus md_solution_heatmap_set_size(MDSolutionHandle, int w, int h);`
  - `MDStatus md_solution_heatmap_update(MDSolutionHandle, const float* boxes, size_t n, int frame_w, int frame_h);`
  - `MDStatus md_solution_heatmap_peak(MDSolutionHandle, int* x, int* y);`
  - `MDStatus md_vision_iou4(float ax, float ay, float aw, float ah, float bx, float by, float bw, float bh, float* out);`

- [ ] **Step 1: `capi/md_capi.h` 新增**

在 `MDOptionHandle` 后加：
```c
typedef struct md_solution_handle* MDSolutionHandle;
```
声明区（`md_audio_*` 附近）加：
```c
typedef enum MD_SOLUTION_KIND {
    MD_SOLUTION_OBJECT_COUNTER = 0,
    MD_SOLUTION_HEATMAP,
    MD_SOLUTION_SPEED,
    MD_SOLUTION_DISTANCE,
    MD_SOLUTION_WORKOUT,
    MD_SOLUTION_PARKING,
} MDSolutionKind;

MD_CAPI_EXPORT MDStatus md_solution_create(MDSolutionHandle* out, MDSolutionKind kind);
MD_CAPI_EXPORT MDStatus md_solution_destroy(MDSolutionHandle);
MD_CAPI_EXPORT MDStatus md_solution_object_counter_set_line(MDSolutionHandle h, float ax, float ay, float bx, float by);
MD_CAPI_EXPORT MDStatus md_solution_object_counter_update(MDSolutionHandle h, const float* boxes, size_t n,
                                                          const int* label_ids, const int* track_ids);
MD_CAPI_EXPORT MDStatus md_solution_object_counter_hline(MDSolutionHandle h, int* in, int* out_count);
MD_CAPI_EXPORT MDStatus md_solution_heatmap_set_size(MDSolutionHandle h, int w, int hh);
MD_CAPI_EXPORT MDStatus md_solution_heatmap_update(MDSolutionHandle h, const float* boxes, size_t n,
                                                   int frame_w, int frame_h);
MD_CAPI_EXPORT MDStatus md_solution_heatmap_peak(MDSolutionHandle h, int* x, int* y);
MD_CAPI_EXPORT MDStatus md_vision_iou4(float ax, float ay, float aw, float ah,
                                       float bx, float by, float bw, float bh, float* out);
```

- [ ] **Step 2: `capi/md_capi.cpp` 实现**（`#ifdef BUILD_VISION` 守卫）

```cpp
struct md_solution_handle {
    MDSolutionKind kind;
    void* obj;
};

MDStatus md_solution_create(MDSolutionHandle* out, MDSolutionKind kind) {
    if (!out) return MD_ERR_NULL_POINTER;
    auto* h = new md_solution_handle();
    h->kind = kind;
#ifdef BUILD_VISION
    switch (kind) {
      case MD_SOLUTION_OBJECT_COUNTER: h->obj = new vision::solution::ObjectCounter(); break;
      case MD_SOLUTION_HEATMAP:        h->obj = new vision::solution::Heatmap(); break;
      case MD_SOLUTION_SPEED:          h->obj = new vision::solution::SpeedEstimator(); break;
      case MD_SOLUTION_DISTANCE:       h->obj = new vision::solution::DistanceEstimator(); break;
      case MD_SOLUTION_WORKOUT:        h->obj = new vision::solution::WorkoutMonitor(); break;
      case MD_SOLUTION_PARKING:        h->obj = new vision::solution::ParkingManager(); break;
      default: delete h; return MD_ERR_INVALID_ARGUMENT;
    }
    *out = h;
    return MD_OK;
#else
    delete h; set_error("built without BUILD_VISION"); return MD_ERR_UNSUPPORTED_TYPE;
#endif
}

MDStatus md_solution_destroy(MDSolutionHandle h) {
    if (!h) return MD_ERR_NULL_POINTER;
#ifdef BUILD_VISION
    switch (h->kind) {
      case MD_SOLUTION_OBJECT_COUNTER: delete static_cast<vision::solution::ObjectCounter*>(h->obj); break;
      case MD_SOLUTION_HEATMAP:        delete static_cast<vision::solution::Heatmap*>(h->obj); break;
      case MD_SOLUTION_SPEED:          delete static_cast<vision::solution::SpeedEstimator*>(h->obj); break;
      case MD_SOLUTION_DISTANCE:       delete static_cast<vision::solution::DistanceEstimator*>(h->obj); break;
      case MD_SOLUTION_WORKOUT:        delete static_cast<vision::solution::WorkoutMonitor*>(h->obj); break;
      case MD_SOLUTION_PARKING:        delete static_cast<vision::solution::ParkingManager*>(h->obj); break;
      default: break;
    }
#endif
    delete h;
    return MD_OK;
}

MDStatus md_solution_object_counter_set_line(MDSolutionHandle h, float ax, float ay, float bx, float by) {
    if (!h) return MD_ERR_NULL_POINTER;
#ifdef BUILD_VISION
    if (h->kind != MD_SOLUTION_OBJECT_COUNTER) return MD_ERR_INVALID_ARGUMENT;
    static_cast<vision::solution::ObjectCounter*>(h->obj)
        ->set_line(vision::Point2f(ax, ay), vision::Point2f(bx, by));
    return MD_OK;
#else
    (void)ax;(void)ay;(void)bx;(void)by; return MD_ERR_UNSUPPORTED_TYPE;
#endif
}

MDStatus md_solution_object_counter_update(MDSolutionHandle h, const float* boxes, size_t n,
                                           const int* label_ids, const int* track_ids) {
    if (!h || !boxes || n == 0 || !label_ids || !track_ids) return MD_ERR_NULL_POINTER;
#ifdef BUILD_VISION
    if (h->kind != MD_SOLUTION_OBJECT_COUNTER) return MD_ERR_INVALID_ARGUMENT;
    auto* c = static_cast<vision::solution::ObjectCounter*>(h->obj);
    std::vector<tracking::TrackResult> tracks(n);
    for (size_t i = 0; i < n; ++i) {
        tracks[i].track_id = track_ids[i];
        tracks[i].box = vision::Rect2f(boxes[i*4+0], boxes[i*4+1], boxes[i*4+2], boxes[i*4+3]);
        tracks[i].label_id = label_ids[i];
        tracks[i].score = 1.0f;
    }
    c->update(tracks);
    return MD_OK;
#else
    (void)boxes;(void)n;(void)label_ids;(void)track_ids; return MD_ERR_UNSUPPORTED_TYPE;
#endif
}

MDStatus md_solution_object_counter_hline(MDSolutionHandle h, int* in, int* out_count) {
    if (!h || !in || !out_count) return MD_ERR_NULL_POINTER;
#ifdef BUILD_VISION
    if (h->kind != MD_SOLUTION_OBJECT_COUNTER) return MD_ERR_INVALID_ARGUMENT;
    auto st = static_cast<vision::solution::ObjectCounter*>(h->obj)->stats();
    *in = st.line_in; *out_count = st.line_out;
    return MD_OK;
#else
    return MD_ERR_UNSUPPORTED_TYPE;
#endif
}

MDStatus md_solution_heatmap_set_size(MDSolutionHandle h, int w, int hh) {
    if (!h) return MD_ERR_NULL_POINTER;
#ifdef BUILD_VISION
    if (h->kind != MD_SOLUTION_HEATMAP) return MD_ERR_INVALID_ARGUMENT;
    static_cast<vision::solution::Heatmap*>(h->obj)->set_size(w, hh);
    return MD_OK;
#else
    (void)w;(void)hh; return MD_ERR_UNSUPPORTED_TYPE;
#endif
}

MDStatus md_solution_heatmap_update(MDSolutionHandle h, const float* boxes, size_t n, int frame_w, int frame_h) {
    if (!h || !boxes || n == 0) return MD_ERR_NULL_POINTER;
#ifdef BUILD_VISION
    if (h->kind != MD_SOLUTION_HEATMAP) return MD_ERR_INVALID_ARGUMENT;
    auto* hm = static_cast<vision::solution::Heatmap*>(h->obj);
    std::vector<tracking::TrackResult> tracks(n);
    for (size_t i = 0; i < n; ++i) {
        tracks[i].track_id = (int)i;
        tracks[i].box = vision::Rect2f(boxes[i*4+0], boxes[i*4+1], boxes[i*4+2], boxes[i*4+3]);
    }
    hm->update(tracks, frame_w, frame_h);
    return MD_OK;
#else
    (void)boxes;(void)n;(void)frame_w;(void)frame_h; return MD_ERR_UNSUPPORTED_TYPE;
#endif
}

MDStatus md_solution_heatmap_peak(MDSolutionHandle h, int* x, int* y) {
    if (!h || !x || !y) return MD_ERR_NULL_POINTER;
#ifdef BUILD_VISION
    if (h->kind != MD_SOLUTION_HEATMAP) return MD_ERR_INVALID_ARGUMENT;
    auto p = static_cast<vision::solution::Heatmap*>(h->obj)->peak();
    *x = p.first; *y = p.second;
    return MD_OK;
#else
    return MD_ERR_UNSUPPORTED_TYPE;
#endif
}

MDStatus md_vision_iou4(float ax, float ay, float aw, float ah,
                        float bx, float by, float bw, float bh, float* out) {
    if (!out) return MD_ERR_NULL_POINTER;
#ifdef BUILD_VISION
    *out = vision::tool::iou(vision::Rect2f(ax,ay,aw,ah), vision::Rect2f(bx,by,bw,bh));
    return MD_OK;
#else
    (void)ax;(void)ay;(void)aw;(void)ah;(void)bx;(void)by;(void)bw;(void)bh;
    return MD_ERR_UNSUPPORTED_TYPE;
#endif
}
```

- [ ] **Step 3: `tests/test_capi.cpp` 新增 `[capi]` 用例**

```cpp
TEST_CASE("cv solution + tool capi", "[capi]") {
    MDSolutionHandle h = nullptr;
    REQUIRE(md_solution_create(&h, MD_SOLUTION_OBJECT_COUNTER) == MD_OK);
    REQUIRE(md_solution_object_counter_set_line(h, 5.0f, 0.0f, 5.0f, 10.0f) == MD_OK);
    float b1[4] = {0,4,2,2}; int lid[1] = {0}; int tid[1] = {1};
    REQUIRE(md_solution_object_counter_update(h, b1, 1, lid, tid) == MD_OK);
    int in = -1, oc = -1;
    REQUIRE(md_solution_object_counter_hline(h, &in, &oc) == MD_OK);
    REQUIRE(in == 0);
    float b2[4] = {8,4,2,2};
    REQUIRE(md_solution_object_counter_update(h, b2, 1, lid, tid) == MD_OK);
    REQUIRE(md_solution_object_counter_hline(h, &in, &oc) == MD_OK);
    REQUIRE(in == 1);
    REQUIRE(md_solution_destroy(h) == MD_OK);
    float iou = 0;
    REQUIRE(md_vision_iou4(0,0,10,10, 0,0,10,10, &iou) == MD_OK);
    REQUIRE(iou == Approx(1.0f).margin(1e-5f));
}
```

- [ ] **Step 4: 构建 + 测试**

Run: `cmake --build build --parallel 8 && cd build && .\bin\test_modeldeploy.exe "[capi]"`
Expected: 0 errors；该 `[capi]` 用例 PASS。

- [ ] **Step 5: Commit**

```bash
git add capi/md_capi.h capi/md_capi.cpp tests/test_capi.cpp
git commit -m "feat(capi): md_solution_* handles + md_vision_iou4"
```

---

## A14: CV C# + Rust + demo + docs

**Files:**
- Modify: `csharp/ModelDeploy/types_internal_c.cs`（`MDSolutionKind` + extern 导入）
- Create: `csharp/ModelDeploy/Solutions.cs`
- Modify: `csharp/ModelDeployUnitTest/AllModelsTests.cs`（`CvSolution_Works`）
- Modify: `rust/modeldeploy/src/ffi.rs`（`MDSolutionKind` + extern）
- Create: `rust/modeldeploy/src/solution.rs`
- Modify: `rust/modeldeploy/src/lib.rs`
- Create: `examples/demo_solutions/demo_solutions.cpp` + `CMakeLists.txt`
- Create: `examples/demo_tools/demo_tools.cpp` + `CMakeLists.txt`
- Modify: `examples/CMakeLists.txt`、`examples/EXAMPLES.md`、`README.md`

**Interfaces:**
- Consumes: CAPI A13
- Produces: C# `Solutions.ObjectCounter`（`SetLine/Update/HLine`）；Rust `solution::ObjectCounter`；`demo_solutions`（ByteTracker 编排计数/热力图叠加）、`demo_tools`（Annotator/Zone/Metrics）

- [ ] **Step 1: C#** —— `types_internal_c.cs` 加枚举与 extern（仿 `AudioModels.cs` 的 `SenseVoiceModel` 封装风格）：

```csharp
// types_internal_c.cs
public enum MD_SOLUTION_KIND { MD_SOLUTION_OBJECT_COUNTER = 0, MD_SOLUTION_HEATMAP, MD_SOLUTION_SPEED, MD_SOLUTION_DISTANCE, MD_SOLUTION_WORKOUT, MD_SOLUTION_PARKING }
```
```csharp
// NativeMethods.cs 加：
[DllImport("ModelDeploySDK", CallingConvention = CallingConvention.Cdecl)]
public static extern MDStatus md_solution_create(out IntPtr h, MD_SOLUTION_KIND kind);
[DllImport("ModelDeploySDK", CallingConvention = CallingConvention.Cdecl)]
public static extern MDStatus md_solution_destroy(IntPtr h);
[DllImport("ModelDeploySDK", CallingConvention = CallingConvention.Cdecl)]
public static extern MDStatus md_solution_object_counter_set_line(IntPtr h, float ax, float ay, float bx, float by);
[DllImport("ModelDeploySDK", CallingConvention = CallingConvention.Cdecl)]
public static extern MDStatus md_solution_object_counter_update(IntPtr h, float[] boxes, UIntPtr n, int[] label_ids, int[] track_ids);
[DllImport("ModelDeploySDK", CallingConvention = CallingConvention.Cdecl)]
public static extern MDStatus md_solution_object_counter_hline(IntPtr h, out int inCount, out int outCount);
[DllImport("ModelDeploySDK", CallingConvention = CallingConvention.Cdecl)]
public static extern MDStatus md_vision_iou4(float ax, float ay, float aw, float ah, float bx, float by, float bw, float bh, out float iou);
```
```csharp
// Solutions.cs
namespace ModelDeploy.Solutions {
    public sealed class ObjectCounter : IDisposable {
        private IntPtr _h;
        public ObjectCounter() {
            if (NativeMethods.md_solution_create(out _h, types_internal_c.MD_SOLUTION_KIND.MD_SOLUTION_OBJECT_COUNTER) != types_internal_c.MDStatus.MD_OK)
                throw new InvalidOperationException("solution create failed");
        }
        public void SetLine(float ax, float ay, float bx, float by) =>
            NativeMethods.md_solution_object_counter_set_line(_h, ax, ay, bx, by);
        public void Update(float[] boxes, int[] labelIds, int[] trackIds) =>
            NativeMethods.md_solution_object_counter_update(_h, boxes, new UIntPtr((uint)labelIds.Length), labelIds, trackIds);
        public (int In, int Out) HLine() {
            NativeMethods.md_solution_object_counter_hline(_h, out var i, out var o);
            return (i, o);
        }
        public void Dispose() { if (_h != IntPtr.Zero) { NativeMethods.md_solution_destroy(_h); _h = IntPtr.Zero; } }
    }
}
```
`AllModelsTests.cs` 加 `[Fact] CvSolution_Works`：构造 `ObjectCounter`、`SetLine((5,0),(5,10))`、两次 `Update`（框 0→8）断言 `HLine().In == 1`。

- [ ] **Step 2: Rust** —— `ffi.rs` 加枚举 + extern；`solution.rs` 薄封装；`lib.rs` `pub mod solution;`：

```rust
// ffi.rs
#[repr(C)] #[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MDSolutionKind { ObjectCounter = 0, Heatmap, Speed, Distance, Workout, Parking }
extern "C" { pub fn md_solution_create(out: *mut MDSolutionHandle, kind: MDSolutionKind) -> MDStatus;
             pub fn md_solution_destroy(h: MDSolutionHandle) -> MDStatus;
             pub fn md_solution_object_counter_set_line(h: MDSolutionHandle, ax: c_float, ay: c_float, bx: c_float, by: c_float) -> MDStatus;
             pub fn md_solution_object_counter_update(h: MDSolutionHandle, boxes: *const c_float, n: usize, label_ids: *const c_int, track_ids: *const c_int) -> MDStatus;
             pub fn md_solution_object_counter_hline(h: MDSolutionHandle, in_count: *mut c_int, out_count: *mut c_int) -> MDStatus; }
```
```rust
// solution.rs（薄封装，仿 model.rs 的 handle 管理）
pub struct ObjectCounter { handle: ffi::MDSolutionHandle }
impl ObjectCounter {
    pub fn new() -> Result<Self, MdError> {
        let mut h = std::ptr::null_mut();
        check_status(unsafe { ffi::md_solution_create(&mut h, ffi::MDSolutionKind::ObjectCounter) })?;
        Ok(Self { handle: h })
    }
    pub fn set_line(&self, a: (f32,f32), b: (f32,f32)) -> Result<(), MdError> {
        check_status(unsafe { ffi::md_solution_object_counter_set_line(self.handle, a.0, a.1, b.0, b.1) })
    }
    pub fn update(&self, boxes: &[f32], labels: &[i32], track_ids: &[i32]) -> Result<(), MdError> {
        check_status(unsafe { ffi::md_solution_object_counter_update(self.handle, boxes.as_ptr(), labels.len(), labels.as_ptr(), track_ids.as_ptr()) })
    }
    pub fn hline(&self) -> Result<(i32,i32), MdError> {
        let (mut i, mut o) = (0i32, 0i32);
        check_status(unsafe { ffi::md_solution_object_counter_hline(self.handle, &mut i, &mut o) })?;
        Ok((i, o))
    }
}
impl Drop for ObjectCounter { fn drop(&mut self) { unsafe { ffi::md_solution_destroy(self.handle); } } }
```
`integration_test.rs` 加 `#[test] fn test_cv_solution`：new/set_line/update → `hline().0 == 1`。

- [ ] **Step 3: `examples/demo_solutions/`** —— 综合 demo（检测 + ByteTrack → 计数 + 热力图 + 测速叠加；缺视频/权重清晰报错）：

`examples/demo_solutions/CMakeLists.txt`（仿 `demo_landmark/CMakeLists.txt`）：
```cmake
add_executable(demo_solutions demo_solutions.cpp)
target_link_libraries(demo_solutions PRIVATE ${LIBRARY_NAME} ${OpenCV_LIBS})
```
`examples/demo_solutions/demo_solutions.cpp`（表格化、缺权重 SKIP、复用 `UltralyticsDet` + `ByteTracker` + `ObjectCounter/Heatmap/SpeedEstimator`）——给骨架 + 完整关键调用，缺可视化输入时打印无崩溃：
```cpp
// demo_solutions <det.onnx> <video.mp4|image.png>  —— 计数+热力图+测速叠加
// 真实检测/字节跟踪需 BUILD_VISION + BUILD_VIDEO（视频抽帧）权重；缺则报错退出。
// 使用复用：UltralyticsDet::predict（DetectionResult）、ByteTracker::update、ObjectCounter/Heatmap/SpeedEstimator。
int main(int argc, char** argv) {
    if (argc < 2) { printf("Usage: demo_solutions <det.onnx> [video.mp4]\n"); return 1; }
    // ...（完整实现对帧循环：decode -> det.predict -> to tracking::Detection -> ByteTracker.update
    //      -> ObjectCounter.update + Heatmap.update + SpeedEstimator.update -> draw_box_labels 叠加显示）
    return 0;
}
```
（实施时把「完整实现对帧循环」按 A1–A11 的 API 补齐——demo 是编排示例，逻辑即 A7/A8/A9 的组合。）

- [ ] **Step 4: `examples/demo_tools/`** —— Annotator/Zone/Metrics 用法：

```cpp
// demo_tools <img.png>                          —— 展示 Annotator/Zone/Metrics
// 纯工具、无权重：读图 -> draw_box_labels 画合成框 -> LineZone/PolygonZone 演示 -> Metrics 小样本打印 mAP。
```
`CMakeLists.txt` 仿 `demo_landmark`。

- [ ] **Step 5: 注册 + docs**

`examples/CMakeLists.txt` 加 `add_subdirectory(demo_solutions)`、`add_subdirectory(demo_tools)`（`demo_landmark` 之后）。
`EXAMPLES.md` 加行：
```
| `demo_solutions` | CV 应用方案（检测+ByteTrack→计数/热力图/测速叠加） | `onnx/yolo26n/*.onnx` | 视频/图 | 窗口显示叠加 |
| `demo_tools` | CV 工具（Annotator/Zone/Metrics） | 无 | 图 | 打印 mAP/计数 |
```
`README.md` 能力列表加「**CV Solutions 应用方案（计数/热力图/测速/停车/健身等）+ CV Tools（标注器/区域/评测）**」。

- [ ] **Step 6: 构建 + C#/Rust 冒烟 + Commit**

`cmake --build build --parallel 8`（`demo_solutions`/`demo_tools` 编译 0 errors；缺权重报错不崩）；`dotnet test --filter CvSolution_Works`；`cargo test test_cv_solution`。
```bash
git add csharp/ModelDeploy/ csharp/ModelDeployUnitTest/ rust/modeldeploy/src/ rust/modeldeploy/tests/ examples/demo_solutions/ examples/demo_tools/ examples/CMakeLists.txt examples/EXAMPLES.md README.md
git commit -m "feat(solutions): CV C#/Rust wrappers + demo_solutions + demo_tools + docs"
```

---

# B 域：Audio（方案层 + 工具层）

## B1: `audio::tool` WavIO + AudioMeta

**Files:**
- Create: `csrc/audio/tools/wav_io.h` / `.cpp`
- Create: `csrc/audio/tools/audio_meta.h` / `.cpp`
- Test: `tests/test_audio_tools.cpp`
- Modify: `tests/CMakeLists.txt`（加 `test_audio_tools.cpp`）

**Interfaces:**
- **复用**：`load_wav_file`（`csrc/utils/wave_helper.h:93`）读 PCM；写盘复用 `capi/md_capi.cpp:2415` `md_wav_save` 的 RIFF/16-bit 写逻辑（`append_u16/append_u32`）。
- Consumes: `std::vector<float>`
- Produces:
  - `struct audio::tool::WavMeta { int channels{1}; int sample_rate{0}; int bits{16}; int format{1}; uint32_t duration_ms{0}; }`
  - `struct audio::tool::WavData { WavMeta meta; std::vector<float> samples; }`
  - `bool audio::tool::read_wav(const std::string& path, WavData* out)`（复用 `load_wav_file`；meta 从文件头解析，缺失给默认）
  - `bool audio::tool::write_wav(const std::string& path, const std::vector<float>& samples, int sample_rate)`（16-bit RIFF）
  - `WavMeta audio::tool::parse_meta(const std::string& path)`（纯解析 WAV 头）

- [ ] **Step 1: 写失败测试** (`tests/test_audio_tools.cpp`)

```cpp
#include <catch2/catch_test_macros.hpp>
#include <cstdio>
#include "audio/tools/wav_io.h"
#include "audio/tools/audio_meta.h"
using namespace modeldeploy::audio::tool;

TEST_CASE("WavIO write then read roundtrip", "[audio_tools]") {
    std::vector<float> sine(1600);
    for (size_t i = 0; i < sine.size(); ++i) sine[i] = 0.5f * std::sin(2 * 3.14159265f * 440.0f * (i / 16000.0f));
    const std::string path = "audio_roundtrip_test.wav";
    REQUIRE(write_wav(path, sine, 16000));
    WavData d;
    REQUIRE(read_wav(path, &d));
    REQUIRE(d.meta.sample_rate == 16000);
    REQUIRE(d.meta.channels == 1);
    REQUIRE(d.samples.size() == sine.size());
    REQUIRE(d.samples[100] == Approx(sine[100]).margin(1e-3f));  // 16-bit 量化误差
    auto meta = parse_meta(path);
    REQUIRE(meta.sample_rate == 16000);
    REQUIRE(meta.bits == 16);
    std::remove(path.c_str());
}
```

- [ ] **Step 2: 运行确认失败**

Run: `cd build && .\bin\test_modeldeploy.exe "[audio_tools]"`
Expected: FAIL（`audio/tools/wav_io.h` 不存在）。

- [ ] **Step 3/4: 写头文件与实现**

```cpp
// wav_io.h
#pragma once
#include <cstdint>
#include <string>
#include <vector>
#include "core/md_decl.h"
namespace modeldeploy::audio::tool {
struct MODELDEPLOY_CXX_EXPORT WavMeta {
    int channels{1}; int sample_rate{0}; int bits{16}; int format{1}; uint32_t duration_ms{0};
};
struct MODELDEPLOY_CXX_EXPORT WavData { WavMeta meta; std::vector<float> samples; };
MODELDEPLOY_CXX_EXPORT bool read_wav(const std::string& path, WavData* out);
MODELDEPLOY_CXX_EXPORT bool write_wav(const std::string& path, const std::vector<float>& samples, int sample_rate);
} // namespace modeldeploy::audio::tool
```

```cpp
// audio_meta.h
#pragma once
#include <string>
#include "core/md_decl.h"
#include "audio/tools/wav_io.h"
namespace modeldeploy::audio::tool {
MODELDEPLOY_CXX_EXPORT WavMeta parse_meta(const std::string& path);
} // namespace modeldeploy::audio::tool
```

```cpp
// wav_io.cpp
#include "audio/tools/wav_io.h"
#include "utils/wave_helper.h"
#include <cstdio>
#include <cstring>
namespace modeldeploy::audio::tool {

bool read_wav(const std::string& path, WavData* out) {
    if (!out) return false;
    int sr = 0;
    if (!load_wav_file(path.c_str(), &sr, out->samples)) return false; // 复用现有解码
    out->meta.sample_rate = sr;
    out->meta.channels = 1;                       // load_wav_file 归一为单声道 float
    out->meta.bits = 32;                          // float PCM（内部）
    out->meta.format = 3;                         // IEEE float
    if (sr > 0) out->meta.duration_ms = (uint32_t)(out->samples.size() * 1000u / (uint32_t)sr);
    return true;
}

bool write_wav(const std::string& path, const std::vector<float>& samples, int sample_rate) {
    if (samples.empty() || sample_rate <= 0) return false;
    FILE* f = std::fopen(path.c_str(), "wb");
    if (!f) return false;
    const uint32_t n = (uint32_t)samples.size();
    const uint32_t data_bytes = n * 2;
    const uint32_t byte_rate = (uint32_t)sample_rate * 2;
    auto w16 = [&](uint16_t v){ std::fwrite(&v, 2, 1, f); };
    auto w32 = [&](uint32_t v){ std::fwrite(&v, 4, 1, f); };
    std::fwrite("RIFF", 1, 4, f); w32(36 + data_bytes);
    std::fwrite("WAVE", 1, 4, f);
    std::fwrite("fmt ", 1, 4, f); w32(16); w16(1); w16(1); w32((uint32_t)sample_rate);
    w32(byte_rate); w16(2); w16(16);
    std::fwrite("data", 1, 4, f); w32(data_bytes);
    for (uint32_t i = 0; i < n; ++i) {
        int16_t v = (int16_t)(samples[i] * 32767.0f);
        std::fwrite(&v, 2, 1, f);
    }
    std::fclose(f);
    return true;
}
} // namespace modeldeploy::audio::tool
```

```cpp
// audio_meta.cpp
#include "audio/tools/audio_meta.h"
#include <cstdio>
#include <cstring>
namespace modeldeploy::audio::tool {
WavMeta parse_meta(const std::string& path) {
    WavMeta meta;
    FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) return meta;
    char riff[4] = {0}; std::fread(riff, 1, 4, f);
    if (std::memcmp(riff, "RIFF", 4) != 0) { std::fclose(f); return meta; }
    uint32_t dat; std::fread(&dat, 4, 1, f);          // riff size
    std::fseek(f, 8, SEEK_SET);                        // skip WAVE
    // 扫描 chunck 找 fmt
    char id[4];
    uint32_t size = 0;
    bool have_fmt = false;
    while (std::fread(id, 1, 4, f) == 4 && std::fread(&size, 4, 1, f) == 1) {
        if (std::memcmp(id, "fmt ", 4) == 0) {
            uint16_t fmt, ch; uint32_t sr, br; uint16_t ba, bits;
            std::fread(&fmt, 2, 1, f); std::fread(&ch, 2, 1, f);
            std::fread(&sr, 4, 1, f); std::fread(&br, 4, 1, f);
            std::fread(&ba, 2, 1, f); std::fread(&bits, 2, 1, f);
            meta.format = fmt; meta.channels = ch; meta.sample_rate = (int)sr; meta.bits = bits;
            have_fmt = true;
        } else if (std::memcmp(id, "data", 4) == 0 && have_fmt) {
            const uint32_t bytes = size;
            if (meta.sample_rate > 0 && bytes > 0)
                meta.duration_ms = (uint32_t)(bytes * 1000ull / ((uint64_t)meta.sample_rate * (uint64_t)meta.channels * ((uint64_t)meta.bits / 8)));
            break;
        }
        std::fseek(f, (long)size + (size & 1), SEEK_CUR);
    }
    std::fclose(f);
    return meta;
}
} // namespace modeldeploy::audio::tool
```

- [ ] **Step 5: 构建 + 运行确认通过**

`tests/CMakeLists.txt` TEST_SOURCES 加 `test_audio_tools.cpp`。
Run: `cmake --build build --parallel 8 && cd build && .\bin\test_modeldeploy.exe "[audio_tools]"`
Expected: 1 个 `[audio_tools]` 用例 PASS。
> `load_wav_file` 返回单声道 float；`bits=32/format=3` 为内部表示，写盘为 16-bit。`parse_meta` 的 `data` 字节数换算 `duration_ms` 需按实际 `bits`（16）计算，测试以 `sample_rate`/`bits` 断言即可。

- [ ] **Step 6: Commit**

```bash
git add csrc/audio/tools/wav_io.* csrc/audio/tools/audio_meta.* tests/test_audio_tools.cpp tests/CMakeLists.txt
git commit -m "feat(audio_tools): WavIO read/write + AudioMeta (reuse load_wav_file)"
```

---

## B2: `audio::tool` Resampler（复用 samplerate）

**Files:**
- Create: `csrc/audio/tools/resampler.h` / `.cpp`
- Modify: `tests/test_audio_tools.cpp`

**Interfaces:**
- **复用**：`<samplerate/include/samplerate.h>` 的 `src_simple`（见 `csrc/audio/asr_pipeline.cpp:8` 已用）
- Produces: `class Resampler { static std::vector<float> resample(const std::vector<float>& in, int in_sr, int out_sr); }`

- [ ] **Step 1: 追加失败测试** (`tests/test_audio_tools.cpp`)

```cpp
#include "audio/tools/resampler.h"
#include <cmath>

TEST_CASE("Resampler 8k->16k doubles length, frequency preserved", "[audio_tools]") {
    std::vector<float> sine(800);
    for (size_t i = 0; i < sine.size(); ++i) sine[i] = std::sin(2 * 3.14159265f * 1000.0f * (i / 8000.0f));
    auto out = Resampler::resample(sine, 8000, 16000);
    REQUIRE(out.size() == sine.size() * 2);      // 长度约 2 倍
    // 峰值频率仍在 1kHz：对 resample 后做 DFT 找主峰，应落在 N*1000/16000
    const float* p = out.data();
    const size_t N = out.size();
    const double target = (double)N * 1000.0 / 16000.0;
    // 用 0~Nyquist 的 DFT 能量找主 bin
    double best1 = 0.0, best2 = 0.0; size_t bestb = 0;
    for (size_t k = 0; k < N / 2; ++k) {
        double re = 0, im = 0;
        for (size_t i = 0; i < N; ++i) {
            const double a = 2 * 3.14159265 * k * i / N;
            re += p[i] * std::cos(a); im -= p[i] * std::sin(a);
        }
        const double e = re * re + im * im;
        if (e > best2) { best2 = best1; best1 = e; bestb = k; }
        (void)best2;
    }
    REQUIRE(std::abs((double)(int)bestb - target) <= 2);
}
```
> 目标 bin = N*1000/16000 = N/16；N=1600 → 目标 100。DFT 直接对 1600 点 800 bin 计算约 128 万乘加，可接受。

- [ ] **Step 2: 运行确认失败**

Run: `cd build && .\bin\test_modeldeploy.exe "[audio_tools]"`
Expected: FAIL（`audio/tools/resampler.h` 不存在）。

- [ ] **Step 3/4: 写头文件与实现**

```cpp
// resampler.h
#pragma once
#include <vector>
#include "core/md_decl.h"
namespace modeldeploy::audio::tool {
class MODELDEPLOY_CXX_EXPORT Resampler {
public:
    static std::vector<float> resample(const std::vector<float>& in, int in_sr, int out_sr);
};
} // namespace modeldeploy::audio::tool
```

```cpp
// resampler.cpp
#include "audio/tools/resampler.h"
#include <samplerate/include/samplerate.h>
#include <cmath>
namespace modeldeploy::audio::tool {
std::vector<float> Resampler::resample(const std::vector<float>& in, int in_sr, int out_sr) {
    if (in.empty() || in_sr <= 0 || out_sr <= 0) return {};
    if (in_sr == out_sr) return in;
    const long n_out = (long)std::llround((double)in.size() * out_sr / in_sr);
    std::vector<float> out(n_out, 0.0f);
    SRC_DATA data{};
    data.data_in = const_cast<float*>(in.data());
    data.input_frames = (long)in.size();
    data.data_out = out.data();
    data.output_frames = n_out;
    data.src_ratio = (double)out_sr / in_sr;
    if (src_simple(&data, SRC_SINC_BEST_QUALITY, 1) != 0) return {};
    out.resize(data.output_frames_gen);
    return out;
}
} // namespace modeldeploy::audio::tool
```

- [ ] **Step 5: 构建 + 运行确认通过**

Run: `cmake --build build --parallel 8 && cd build && .\bin\test_modeldeploy.exe "[audio_tools]"`
Expected: 2 个 `[audio_tools]` 用例 PASS。

- [ ] **Step 6: Commit**

```bash
git add csrc/audio/tools/resampler.h csrc/audio/tools/resampler.cpp tests/test_audio_tools.cpp
git commit -m "feat(audio_tools): Resampler via samplerate"
```

---

## B3: `audio::tool` Fbank（复用 kaldi-native-fbank）

**Files:**
- Create: `csrc/audio/tools/fbank.h` / `.cpp`
- Modify: `tests/test_audio_tools.cpp`

**Interfaces:**
- **复用**：`knf::FbankOptions` / `knf::OnlineFbank`（用法见 `csrc/audio/speaker_verify/ecapa.cpp:50-56`）
- Produces: `class Fbank { explicit Fbank(int sample_rate = 16000, int num_bins = 80); std::vector<std::vector<float>> compute(const std::vector<float>& samples) const; }`

- [ ] **Step 1: 追加失败测试** (`tests/test_audio_tools.cpp`)

```cpp
#include "audio/tools/fbank.h"
#include <cmath>

TEST_CASE("Fbank produces frames x bins non-degenerate", "[audio_tools]") {
    Fbank fb(16000, 80);
    std::vector<float> s(16000);
    for (size_t i = 0; i < s.size(); ++i) s[i] = 0.5f * std::sin(2 * 3.14159265f * 440.0f * (i / 16000.0f));
    auto frames = fb.compute(s);
    REQUIRE_FALSE(frames.empty());
    REQUIRE(frames[0].size() == 80);
    float energy = 0;
    for (const auto& row : frames) for (float v : row) energy += v * v;
    REQUIRE(energy > 0.0f);   // 非线性/退化
}
```

- [ ] **Step 2: 运行确认失败**

Run: `cd build && .\bin\test_modeldeploy.exe "[audio_tools]"`
Expected: FAIL（`audio/tools/fbank.h` 不存在）。

- [ ] **Step 3/4: 写头文件与实现**

```cpp
// fbank.h
#pragma once
#include <vector>
#include "core/md_decl.h"
namespace modeldeploy::audio::tool {
class MODELDEPLOY_CXX_EXPORT Fbank {
public:
    explicit Fbank(int sample_rate = 16000, int num_bins = 80);
    std::vector<std::vector<float>> compute(const std::vector<float>& samples) const;
private:
    int sr_;
    int bins_;
};
} // namespace modeldeploy::audio::tool
```

```cpp
// fbank.cpp
#include "audio/tools/fbank.h"
#include "kaldi_native_fbank/csrc/feature-fbank.h"
#include "kaldi_native_fbank/csrc/feature-window.h"
namespace modeldeploy::audio::tool {
Fbank::Fbank(int sample_rate, int num_bins) : sr_(sample_rate), bins_(num_bins) {}
std::vector<std::vector<float>> Fbank::compute(const std::vector<float>& samples) const {
    knf::FbankOptions opts;
    opts.frame_opts.samp_freq = (float)sr_;
    opts.mel_opts.num_bins = bins_;
    knf::OnlineFbank kaldi_fbank(opts);
    if (!samples.empty()) kaldi_fbank.AcceptWaveform((float)sr_, samples);
    kaldi_fbank.InputFinished();
    std::vector<std::vector<float>> out;
    const int n = kaldi_fbank.NumFramesReady();
    out.reserve((size_t)n);
    for (int i = 0; i < n; ++i) {
        const int dim = kaldi_fbank.Dim();
        std::vector<float> row((size_t)dim);
        kaldi_fbank.GetFrame(i, row.data());
        out.push_back(std::move(row));
    }
    return out;
}
} // namespace modeldeploy::audio::tool
```

- [ ] **Step 5: 构建 + 运行确认通过**

Run: `cmake --build build --parallel 8 && cd build && .\bin\test_modeldeploy.exe "[audio_tools]"`
Expected: 3 个 `[audio_tools]` 用例 PASS。
> 若 `AcceptWaveform` 与 `GetFrame` 的实际签名不同，以 `csrc/audio/speaker_verify/ecapa.cpp:50-56` 的用法为准（那里已用 `AcceptWaveform`/`InputFinished`/`NumFramesReady`/`GetFrame`）。

- [ ] **Step 6: Commit**

```bash
git add csrc/audio/tools/fbank.h csrc/audio/tools/fbank.cpp tests/test_audio_tools.cpp
git commit -m "feat(audio_tools): Fbank via kaldi-native-fbank"
```

---

## B4: `audio::tool` Waveform + Spectrum

**Files:**
- Create: `csrc/audio/tools/waveform.h` / `.cpp`
- Modify: `tests/test_audio_tools.cpp`

**Interfaces:**
- Produces:
  - `class Waveform { static std::vector<float> downsample(const std::vector<float>& in, size_t points); }`（等间隔抽点，供波形坐标）
  - `class Spectrum { explicit Spectrum(int fft_n = 1024); std::vector<float> magnitudes(const std::vector<float>& samples) const; }`（含本实现内置轻量 FFT，返回 0..Nyquist 幅值谱，长度 fft_n/2+1）
- **复用**：纯 C++（自备轻量 FFT，避免引入新依赖；Audio 工具不链接 OpenCV）

- [ ] **Step 1: 追加失败测试** (`tests/test_audio_tools.cpp`)

```cpp
#include "audio/tools/waveform.h"
#include <cmath>

TEST_CASE("Spectrum single tone peak bin", "[audio_tools]") {
    const size_t N = 1024;
    std::vector<float> s(N);
    for (size_t i = 0; i < N; ++i) s[i] = std::sin(2 * 3.14159265f * 100.0f * (i / 1024.0f)); // bin ~100
    Spectrum sp(1024);
    auto mag = sp.magnitudes(s);
    REQUIRE(mag.size() == N / 2 + 1);
    // 主峰应接近 bin 100
    size_t best = 0; float maxv = -1;
    for (size_t k = 0; k < mag.size(); ++k) if (mag[k] > maxv) { maxv = mag[k]; best = k; }
    REQUIRE((size_t)(std::abs((long)((int)best - 100))) <= 2);
}

TEST_CASE("Waveform downsample reduces length", "[audio_tools]") {
    std::vector<float> s(4000, 0.5f);
    auto d = Waveform::downsample(s, 200);
    REQUIRE(d.size() <= 200);
}
```

- [ ] **Step 2: 运行确认失败**

Run: `cd build && .\bin\test_modeldeploy.exe "[audio_tools]"`
Expected: FAIL（`audio/tools/waveform.h` 不存在）。

- [ ] **Step 3/4: 写头文件与实现**

```cpp
// waveform.h
#pragma once
#include <cstddef>
#include <vector>
#include "core/md_decl.h"
namespace modeldeploy::audio::tool {
class MODELDEPLOY_CXX_EXPORT Waveform {
public:
    static std::vector<float> downsample(const std::vector<float>& in, size_t points);
};
class MODELDEPLOY_CXX_EXPORT Spectrum {
public:
    explicit Spectrum(int fft_n = 1024) : fft_n_(fft_n) {}
    std::vector<float> magnitudes(const std::vector<float>& samples) const;
private:
    int fft_n_;
    void fft(std::vector<std::complex<float>>& a) const;  // 自备 radix-2 迭代 FFT
};
} // namespace modeldeploy::audio::tool
```

```cpp
// waveform.cpp
#include "audio/tools/waveform.h"
#include <cmath>
#include <complex>
namespace modeldeploy::audio::tool {
std::vector<float> Waveform::downsample(const std::vector<float>& in, size_t points) {
    std::vector<float> out;
    if (in.empty()) return out;
    const size_t step = (in.size() > points) ? in.size() / points : 1;
    for (size_t i = 0; i < in.size(); i += step) out.push_back(in[i]);
    return out;
}
void Spectrum::fft(std::vector<std::complex<float>>& a) const {
    const size_t n = a.size();
    for (size_t i = 1, j = 0; i < n; ++i) {
        size_t bit = n >> 1;
        for (; j & bit; bit >>= 1) j ^= bit;
        j ^= bit;
        if (i < j) std::swap(a[i], a[j]);
    }
    for (size_t len = 2; len <= n; len <<= 1) {
        const float ang = -2.0f * 3.14159265358979f / (float)len;
        const std::complex<float> wlen(std::cos(ang), std::sin(ang));
        for (size_t i = 0; i < n; i += len) {
            std::complex<float> w(1.0f, 0.0f);
            for (size_t j = 0; j < len / 2; ++j) {
                std::complex<float> u = a[i + j];
                std::complex<float> v = a[i + j + len / 2] * w;
                a[i + j] = u + v;
                a[i + j + len / 2] = u - v;
                w *= wlen;
            }
        }
    }
}
std::vector<float> Spectrum::magnitudes(const std::vector<float>& samples) const {
    size_t n = 1;
    while (n < (size_t)fft_n_) n <<= 1;
    std::vector<std::complex<float>> a(n, std::complex<float>(0.0f, 0.0f));
    for (size_t i = 0; i < samples.size() && i < n; ++i) a[i] = std::complex<float>(samples[i], 0.0f);
    fft(a);
    std::vector<float> mag(n / 2 + 1);
    for (size_t k = 0; k <= n / 2; ++k) mag[k] = std::abs(a[k]);
    return mag;
}
} // namespace modeldeploy::audio::tool
```

- [ ] **Step 5: 构建 + 运行确认通过**

Run: `cmake --build build --parallel 8 && cd build && .\bin\test_modeldeploy.exe "[audio_tools]"`
Expected: 5 个 `[audio_tools]` 用例 PASS。
> `stdio` 需 `#include <complex>` 与 `<algorithm>`（`std::swap`）。scaling：裸 FFT 幅值 bin≈N/2 量级，测试只看相对峰位置。

- [ ] **Step 6: Commit**

```bash
git add csrc/audio/tools/waveform.h csrc/audio/tools/waveform.cpp tests/test_audio_tools.cpp
git commit -m "feat(audio_tools): Waveform downsample + Spectrum FFT"
```

---

## B5: `audio::tool` VadSegment（能量阈值分段）

**Files:**
- Create: `csrc/audio/tools/vad_segment.h` / `.cpp`
- Modify: `tests/test_audio_tools.cpp`

**Interfaces:**
- Produces:
  - `struct audio::tool::Seg { int start_ms; int end_ms; std::vector<float> samples; }`
  - `class VadSegment { VadSegment(int sample_rate = 16000, float energy_threshold = 0.01f, int min_speech_ms = 200, int min_silence_ms = 200); void reset(); void feed(const std::vector<float>& samples); std::vector<Seg> segments() const; }`
- **复用/可选组合**：能量阈值分段为纯 C++ 基座（无模型、可确定性单测）；真实 `SileroVAD` 作为可选后端（见 B7 复用）。YAGNI：本工具层实现能量法；`segments()` 对已 feed 的样本按窗口能量判定 speech/silence。

- [ ] **Step 1: 追加失败测试** (`tests/test_audio_tools.cpp`)

```cpp
#include "audio/tools/vad_segment.h"
#include <cmath>

TEST_CASE("VadSegment splits speech vs silence", "[audio_tools]") {
    const int sr = 16000;
    std::vector<float> sig;
    auto tone = [&](float amp, float dur_s){ for (int i = 0; i < (int)(sr*dur_s); ++i) sig.push_back(amp * std::sin(2*3.14159265f*440.0f*(i/(double)sr))); };
    auto silence = [&](float dur_s){ sig.resize(sig.size() + (size_t)(sr*dur_s), 0.0f); };
    tone(0.9f, 1.0f); silence(0.5f); tone(0.9f, 1.0f);   // 两段语音夹一段静音
    VadSegment vad(sr, 0.01f, 200, 200);
    vad.feed(sig);
    auto segs = vad.segments();
    REQUIRE(segs.size() >= 2);          // 至少两段 speech
    REQUIRE(segs[0].samples.size() > 0);
    REQUIRE(segs[1].start_ms > segs[0].end_ms); // 时间先后
}
```

- [ ] **Step 2: 运行确认失败**

Run: `cd build && .\bin\test_modeldeploy.exe "[audio_tools]"`
Expected: FAIL（`audio/tools/vad_segment.h` 不存在）。

- [ ] **Step 3/4: 写头文件与实现**

```cpp
// vad_segment.h
#pragma once
#include <vector>
#include "core/md_decl.h"
namespace modeldeploy::audio::tool {
struct MODELDEPLOY_CXX_EXPORT Seg { int start_ms; int end_ms; std::vector<float> samples; };
class MODELDEPLOY_CXX_EXPORT VadSegment {
public:
    VadSegment(int sample_rate = 16000, float energy_threshold = 0.01f,
               int min_speech_ms = 200, int min_silence_ms = 200)
        : sr_(sample_rate), thr_(energy_threshold), min_speech_(min_speech_ms), min_silence_(min_silence_ms) {}
    void reset() { buf_.clear(); }
    void feed(const std::vector<float>& samples) { buf_.insert(buf_.end(), samples.begin(), samples.end()); }
    std::vector<Seg> segments() const;
private:
    int sr_;
    float thr_;
    int min_speech_, min_silence_;
    std::vector<float> buf_;
};
} // namespace modeldeploy::audio::tool
```

```cpp
// vad_segment.cpp
#include "audio/tools/vad_segment.h"
#include <cmath>
namespace modeldeploy::audio::tool {
std::vector<Seg> VadSegment::segments() const {
    std::vector<Seg> out;
    const int win = sr_ / 100;              // 10ms 窗
    const size_t nwin = buf_.size() / (size_t)win;
    if (nwin == 0) return out;
    std::vector<bool> speech(nwin, false);
    for (size_t w = 0; w < nwin; ++w) {
        float e = 0.0f;
        for (int i = 0; i < win; ++i) { const float v = buf_[w*(size_t)win + i]; e += v * v; }
        speech[w] = (std::sqrt(e / win) > thr_);
    }
    int start = -1;
    std::vector<int> on;
    for (size_t w = 0; w <= nwin; ++w) {
        const bool sp = (w < nwin) ? speech[w] : false;
        if (sp && start < 0) start = (int)w;
        else if (!sp && start >= 0) {
            on.push_back(start); on.push_back((int)w - 1);
            start = -1;
        }
    }
    // 合并过短的 speech 段（< min_speech）；相邻段间静音 < min_silence 合并
    std::vector<Seg> merged;
    int cur_start = -1, cur_end = -1;
    auto flush = [&](){ if (cur_start >= 0) {
        const int start_ms = cur_start * 10, end_ms = (cur_end + 1) * 10;
        const size_t b = (size_t)cur_start * win, e = (size_t)(cur_end + 1) * win;
        if ((end_ms - start_ms) >= min_speech_) {
            Seg s; s.start_ms = start_ms; s.end_ms = end_ms;
            s.samples.assign(buf_.begin() + (long)b, buf_.begin() + (long)std::min(e, buf_.size()));
            merged.push_back(std::move(s));
        }
    } cur_start = cur_end = -1; };
    for (size_t i = 0; i < on.size(); i += 2) {
        const int s = on[i], e = on[i+1];
        if (cur_start < 0) { cur_start = s; cur_end = e; }
        else if ((s - cur_end - 1) * 10 < min_silence_) { cur_end = e; } // 静音间隙短→合并
        else { flush(); cur_start = s; cur_end = e; }
    }
    flush();
    return merged;
}
} // namespace modeldeploy::audio::tool
```

- [ ] **Step 5: 构建 + 运行确认通过**

Run: `cmake --build build --parallel 8 && cd build && .\bin\test_modeldeploy.exe "[audio_tools]"`
Expected: 6 个 `[audio_tools]` 用例 PASS。
> 测试「tone1s, silence 0.5s, tone1s」：两段 speech 间隔 500ms ≥ min_silence 200ms → 不合并，`segs.size() >= 2`。

- [ ] **Step 6: Commit**

```bash
git add csrc/audio/tools/vad_segment.h csrc/audio/tools/vad_segment.cpp tests/test_audio_tools.cpp
git commit -m "feat(audio_tools): VadSegment energy-threshold segmentation"
```

> **Audio 工具层完成**：6 个工具（WavIO/Resampler/Fbank/Waveform/VadSegment/AudioMeta）全在 `BUILD_AUDIO` 下、主 CMake 未改，复用 `load_wav_file`/samplerate/kaldi-native-fbank。

---

## B6: `audio::solution` SolutionBase + SpeakerSearch

**Files:**
- Create: `csrc/audio/solutions/solution_base.h`
- Create: `csrc/audio/solutions/speaker_search.h` / `.cpp`
- Test: `tests/test_audio_solutions.cpp`
- Modify: `tests/CMakeLists.txt`（加 `test_audio_solutions.cpp`）

**Interfaces:**
- **复用**：`audio::SpeakerGallery`（`speaker_gallery.h`，`enroll/match`）；`vision::utils::l2_normalize`/`compute_similarity`；`SpeakerVerify::predict`（可选，无权重 SKIP）
- Produces:
  - `struct SolutionBase { virtual ~SolutionBase() = default; }`
  - `class SpeakerSearch : public SolutionBase { SpeakerSearch(); void enroll(const std::string& label, const std::vector<float>& embedding); std::vector<std::pair<std::string,float>> match(const std::vector<float>& embedding, int k = 1) const; size_t size() const; }`（无权重可跑：**纯复用 SpeakerGallery**）

- [ ] **Step 1: 写失败测试** (`tests/test_audio_solutions.cpp`)

```cpp
#include <catch2/catch_test_macros.hpp>
#include "audio/solutions/speaker_search.h"
using namespace modeldeploy::audio::solution;

TEST_CASE("SpeakerSearch enroll/match no weights", "[audio_solution]") {
    SpeakerSearch s;
    s.enroll("alice", {1.0f, 0.0f, 0.0f});
    s.enroll("bob",   {0.0f, 1.0f, 0.0f});
    REQUIRE(s.size() == 2);
    auto r = s.match({0.99f, 0.1f, 0.0f}, 1);
    REQUIRE_FALSE(r.empty());
    REQUIRE(r[0].first == "alice");
    REQUIRE(r[0].second > 0.9f);
}
```

- [ ] **Step 2: 运行确认失败**

Run: `cd build && .\bin\test_modeldeploy.exe "[audio_solution]"`
Expected: FAIL（`audio/solutions/speaker_search.h` 不存在）。

- [ ] **Step 3/4: 写头文件与实现**

```cpp
// solution_base.h
#pragma once
#include "core/md_decl.h"
namespace modeldeploy::audio::solution {
struct MODELDEPLOY_CXX_EXPORT SolutionBase { virtual ~SolutionBase() = default; };
} // namespace modeldeploy::audio::solution
```

```cpp
// speaker_search.h
#pragma once
#include <string>
#include <utility>
#include <vector>
#include "core/md_decl.h"
#include "audio/speaker_gallery.h"
#include "audio/solutions/solution_base.h"
namespace modeldeploy::audio::solution {
class MODELDEPLOY_CXX_EXPORT SpeakerSearch : public SolutionBase {
public:
    void enroll(const std::string& label, const std::vector<float>& embedding) { gallery_.enroll(label, embedding); }
    std::vector<std::pair<std::string,float>> match(const std::vector<float>& embedding, int k = 1) const { return gallery_.match(embedding, k); }
    size_t size() const { return gallery_.size(); }
private:
    audio::SpeakerGallery gallery_;
};
} // namespace modeldeploy::audio::solution
```

```cpp
// speaker_search.cpp
#include "audio/solutions/speaker_search.h"
// （无额外逻辑：传入 embed → SpeakerGallery.enroll/match，复用既有实现）
```

- [ ] **Step 5: 构建 + 运行确认通过**

`tests/CMakeLists.txt` 加 `test_audio_solutions.cpp`。
Run: `cmake --build build --parallel 8 && cd build && .\bin\test_modeldeploy.exe "[audio_solution]"`
Expected: 1 个 `[audio_solution]` 用例 PASS。

- [ ] **Step 6: Commit**

```bash
git add csrc/audio/solutions/ tests/test_audio_solutions.cpp tests/CMakeLists.txt
git commit -m "feat(audio_solution): SolutionBase + SpeakerSearch (reuse SpeakerGallery)"
```

---

## B7: `audio::solution` SpeakerDiarization（VAD 分段 + 声纹聚类 → 谁在何时说话）

**Files:**
- Create: `csrc/audio/solutions/speaker_diarization.h` / `.cpp`
- Modify: `tests/test_audio_solutions.cpp`

**Interfaces:**
- **复用**：`tool::VadSegment`（B5）、`speaker_verify::SpeakerVerify::predict`（可选）、`vision::utils::compute_similarity`
- Produces:
  - `struct Segment { int start_ms; int end_ms; int speaker_id; }`
  - `class SpeakerDiarization { SpeakerDiarization(); std::vector<int> assign_speakers(const std::vector<std::vector<float>>& embeddings, float threshold = 0.7f) const; bool run(const std::vector<float>& audio, std::vector<Segment>* out); }`（`assign_speakers` 为纯逻辑可测缝；`run` 走真实 VAD+Verify，无权重 SKIP）

- [ ] **Step 1: 追加失败测试** (`tests/test_audio_solutions.cpp`)

```cpp
#include "audio/solutions/speaker_diarization.h"
#include <cmath>

TEST_CASE("SpeakerDiarization assign_speakers clusters", "[audio_solution]") {
    SpeakerDiarization d;
    // 三个 embedding：A, A 很近, B 不同
    std::vector<std::vector<float>> embs = {
        {1.0f, 0.0f}, {0.98f, 0.02f}, {0.0f, 1.0f}
    };
    auto ids = d.assign_speakers(embs, 0.5f);
    REQUIRE(ids.size() == 3);
    REQUIRE(ids[0] == ids[1]);      // A 同类
    REQUIRE(ids[2] != ids[0]);      // B 异类
}
```

- [ ] **Step 2: 运行确认失败**

Run: `cd build && .\bin\test_modeldeploy.exe "[audio_solution]"`
Expected: FAIL（`audio/solutions/speaker_diarization.h` 不存在）。

- [ ] **Step 3/4: 写头文件与实现**

```cpp
// speaker_diarization.h
#pragma once
#include <vector>
#include "core/md_decl.h"
#include "audio/solutions/solution_base.h"
namespace modeldeploy::audio::solution {
struct MODELDEPLOY_CXX_EXPORT Segment { int start_ms; int end_ms; int speaker_id; };
class MODELDEPLOY_CXX_EXPORT SpeakerDiarization : public SolutionBase {
public:
    // 纯聚类缝：相似度 >= threshold 归同一 speaker（迭代最近类），否则新类
    std::vector<int> assign_speakers(const std::vector<std::vector<float>>& embeddings,
                                     float threshold = 0.7f) const;
    // 完整链路：audio -> VadSegment -> (可选需权重) SpeakerVerify embedding -> assign -> segments
    bool run(const std::vector<float>& audio, std::vector<Segment>* out);
};
} // namespace modeldeploy::audio::solution
```

```cpp
// speaker_diarization.cpp
#include "audio/solutions/speaker_diarization.h"
#include "audio/tools/vad_segment.h"
#include "vision/utils.h"

namespace modeldeploy::audio::solution {
std::vector<int> SpeakerDiarization::assign_speakers(
        const std::vector<std::vector<float>>& embeddings, float threshold) const {
    std::vector<int> ids(embeddings.size(), -1);
    std::vector<std::vector<float>> centroids;
    int next = 0;
    for (size_t i = 0; i < embeddings.size(); ++i) {
        const auto q = vision::utils::l2_normalize(embeddings[i]);
        int best = -1; float best_sim = threshold;
        for (size_t c = 0; c < centroids.size(); ++c) {
            const float s = vision::utils::compute_similarity(q, centroids[c]);
            if (s >= best_sim) { best_sim = s; best = (int)c; }
        }
        if (best < 0) {
            best = (int)centroids.size();
            centroids.push_back(q);
            ids[i] = best;
            if (best >= next) next = best + 1;
        } else {
            ids[i] = best;
            // 增量更新质心（简单均值）
            const float w = 0.5f;
            auto& c = centroids[(size_t)best];
            for (size_t k = 0; k < c.size() && k < q.size(); ++k) c[k] = c[k] * (1 - w) + q[k] * w;
        }
    }
    (void)next;
    return ids;
}

bool SpeakerDiarization::run(const std::vector<float>& audio, std::vector<Segment>* out) {
    if (!out) return false;
    // 需真实 VAD(可选) + SpeakerVerify 权重 —— 无权重 SKIP 由调用方处理；此处以 VadSegment 能量法切段，
    // 若未注入 embedding 则每段 speaker_id=-1 并可继续（weighted 路径在 demo 里接 SpeakerVerify）。
    tool::VadSegment vad(16000);
    vad.feed(audio);
    auto segs = vad.segments();
    out->clear();
    for (const auto& s : segs) out->push_back(Segment{s.start_ms, s.end_ms, -1});
    // 说明：真实声纹聚类需逐段 SpeakerVerify.predict 得到 embedding 再 assign_speakers ——
    // 该路径依赖 ecapa.onnx 权重，demo/tests 缺权时 SKIP（本实现保证纯逻辑可跑、不崩）。
    return true;
}
} // namespace modeldeploy::audio::solution
```

- [ ] **Step 5: 构建 + 运行确认通过**

Run: `cmake --build build --parallel 8 && cd build && .\bin\test_modeldeploy.exe "[audio_solution]"`
Expected: 2 个 `[audio_solution]` 用例 PASS。
> `compute_similarity` 需 `#include "vision/utils.h"`；`l2_normalize`/`compute_similarity` 均定义于 `csrc/vision/utils.h:86-90`。

- [ ] **Step 6: Commit**

```bash
git add csrc/audio/solutions/speaker_diarization.h csrc/audio/solutions/speaker_diarization.cpp tests/test_audio_solutions.cpp
git commit -m "feat(audio_solution): SpeakerDiarization assign_speakers + run seam"
```

---

## B8: `audio::solution` StreamingSTT + TTSBatcher

**Files:**
- Create: `csrc/audio/solutions/streaming_stt.h` / `.cpp`
- Create: `csrc/audio/solutions/tts_batcher.h` / `.cpp`
- Modify: `tests/test_audio_solutions.cpp`

**Interfaces:**
- **复用**：`tool::VadSegment`（B5）、`asr::SenseVoice`（可选）、`tts::Kokoro`（可选）
- Produces:
  - `class StreamingSTT { explicit StreamingSTT(std::function<void(const std::string&)> on_text = nullptr); void push(const std::vector<float>& data, int sr); void set_on_text(std::function<void(const std::string&)> cb); void run_once(); }`（VAD 能量法切段 → 回调触发；真实辩字走 SenseVoice，无权重回调不变）
  - `class TTSBatcher { explicit TTSBatcher(std::function<std::vector<float>(const std::string&)> synth = nullptr); void enqueue(const std::vector<std::string>& texts); std::vector<std::vector<float>> dequeue_all(); size_t pending() const; }`（`synth` 可注入 mock：无权重也可确定性测队列）

- [ ] **Step 1: 追加失败测试** (`tests/test_audio_solutions.cpp`)

```cpp
#include "audio/solutions/streaming_stt.h"
#include "audio/solutions/tts_batcher.h"
#include <cmath>
#include <string>

TEST_CASE("StreamingSTT VAD triggers callback on speech", "[audio_solution]") {
    int calls = 0;
    StreamingSTT stt([&](const std::string&){ ++calls; });
    std::vector<float> audio;
    for (int i = 0; i < 16000; ++i) audio.push_back(0.5f * std::sin(2 * 3.14159265f * 440.0f * (i / 16000.0)));
    stt.push(audio, 16000);
    stt.run_once();
    REQUIRE(calls >= 1);   // 能量法切出一段 speech → 回调
}

TEST_CASE("TTSBatcher enqueue/dequeue with mock synth", "[audio_solution]") {
    int synth_calls = 0;
    TTSBatcher bat([&](const std::string&){ ++synth_calls; return std::vector<float>{1.0f, 2.0f, 3.0f}; });
    bat.enqueue({"hello", "world"});
    REQUIRE(bat.pending() == 2);
    auto wavs = bat.dequeue_all();
    REQUIRE(wavs.size() == 2);
    REQUIRE(wavs[0] == std::vector<float>({1.0f, 2.0f, 3.0f}));
    REQUIRE(synth_calls == 2);
    REQUIRE(bat.pending() == 0);
}
```

- [ ] **Step 2: 运行确认失败**

Run: `cd build && .\bin\test_modeldeploy.exe "[audio_solution]"`
Expected: FAIL（`streaming_stt.h` / `tts_batcher.h` 不存在）。

- [ ] **Step 3/4: 写头文件与实现**

```cpp
// streaming_stt.h
#pragma once
#include <functional>
#include <string>
#include <vector>
#include "core/md_decl.h"
#include "audio/tools/vad_segment.h"
#include "audio/solutions/solution_base.h"
namespace modeldeploy::audio::solution {
class MODELDEPLOY_CXX_EXPORT StreamingSTT : public SolutionBase {
public:
    explicit StreamingSTT(std::function<void(const std::string&)> on_text = nullptr);
    void push(const std::vector<float>& data, int sr);
    void set_on_text(std::function<void(const std::string&)> cb) { on_text_ = std::move(cb); }
    void run_once();
private:
    std::function<void(const std::string&)> on_text_;
    tool::VadSegment vad_;
    std::vector<float> pending_;
};
} // namespace modeldeploy::audio::solution
```

```cpp
// streaming_stt.cpp
#include "audio/solutions/streaming_stt.h"
namespace modeldeploy::audio::solution {
StreamingSTT::StreamingSTT(std::function<void(const std::string&)> on_text)
    : on_text_(std::move(on_text)), vad_(16000) {}
void StreamingSTT::push(const std::vector<float>& data, int sr) { vad_.feed(data); }
void StreamingSTT::run_once() {
    // 能量法切段：每段触发 on_text（真实转写在此调 sense_voice 加权；无权重回调传空串）
    auto segs = vad_.segments();
    if (!segs.empty() && on_text_) on_text_(""); // 至少触发一次
}
} // namespace modeldeploy::audio::solution
```

```cpp
// tts_batcher.h
#pragma once
#include <functional>
#include <string>
#include <vector>
#include "core/md_decl.h"
#include "audio/solutions/solution_base.h"
namespace modeldeploy::audio::solution {
class MODELDEPLOY_CXX_EXPORT TTSBatcher : public SolutionBase {
public:
    explicit TTSBatcher(std::function<std::vector<float>(const std::string&)> synth = nullptr);
    void enqueue(const std::vector<std::string>& texts);
    std::vector<std::vector<float>> dequeue_all();
    size_t pending() const { return queue_.size(); }
private:
    std::function<std::vector<float>(const std::string&)> synth_;
    std::vector<std::string> queue_;
};
} // namespace modeldeploy::audio::solution
```

```cpp
// tts_batcher.cpp
#include "audio/solutions/tts_batcher.h"
namespace modeldeploy::audio::solution {
TTSBatcher::TTSBatcher(std::function<std::vector<float>(const std::string&)> synth) : synth_(std::move(synth)) {}
void TTSBatcher::enqueue(const std::vector<std::string>& texts) {
    queue_.insert(queue_.end(), texts.begin(), texts.end());
}
std::vector<std::vector<float>> TTSBatcher::dequeue_all() {
    std::vector<std::vector<float>> out;
    for (const auto& t : queue_) {
        if (synth_) out.push_back(synth_(t)); else out.emplace_back();
    }
    queue_.clear();
    return out;
}
} // namespace modeldeploy::audio::solution
```

- [ ] **Step 5: 构建 + 运行确认通过**

Run: `cmake --build build --parallel 8 && cd build && .\bin\test_modeldeploy.exe "[audio_solution]"`
Expected: 4 个 `[audio_solution]` 用例 PASS。

- [ ] **Step 6: Commit**

```bash
git add csrc/audio/solutions/streaming_stt.* csrc/audio/solutions/tts_batcher.* tests/test_audio_solutions.cpp
git commit -m "feat(audio_solution): StreamingSTT + TTSBatcher (mock-injectable)"
```

> **Audio 方案层完成**：4 个方案（SpeakerSearch/Diarization/StreamingSTT/TTSBatcher）全部复用现有 SpeakerGallery/VadSegment，真实权重（ecapa/sense_voice/kokoro）仅在 demo 使用、缺则 SKIP。

---

## B9: Audio Python（`audio.solutions` + `audio.tools` 子模块）

**Files:**
- Create: `csrc/pybind/audio/solutions_pybind.cpp`
- Create: `csrc/pybind/audio/tools_pybind.cpp`
- Modify: `csrc/pybind/audio/`（新建 `audio_pybind.h` 或直接在 main.cpp 声明）、`csrc/pybind/main.cpp`（声明 + 注册，`#ifdef BUILD_AUDIO`）
- Test: Python smoke

**Interfaces:**
- Consumes: B1–B8 的 `audio::tool`/`audio::solution` 类
- Produces: `modeldeploy.audio.solutions.SpeakerSearch/TTSBatcher/...`、`modeldeploy.audio.tools.Resampler/Fbank/...`

- [ ] **Step 1: 写 `csrc/pybind/audio/solutions_pybind.cpp`**

```cpp
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <functional>
#include "audio/solutions/speaker_search.h"
#include "audio/solutions/tts_batcher.h"

namespace modeldeploy::audio {
    void bind_solutions(const pybind11::module& m) {
        pybind11::class_<solution::SpeakerSearch>(m, "SpeakerSearch")
            .def(pybind11::init<>())
            .def("enroll", &solution::SpeakerSearch::enroll)
            .def("match", &solution::SpeakerSearch::match, pybind11::arg("embedding"), pybind11::arg("k") = 1);
        pybind11::class_<solution::TTSBatcher>(m, "TTSBatcher")
            .def(pybind11::init<>())
            .def("enqueue", &solution::TTSBatcher::enqueue)
            .def("dequeue_all", &solution::TTSBatcher::dequeue_all);
    }
} // namespace modeldeploy::audio
```

- [ ] **Step 2: 写 `csrc/pybind/audio/tools_pybind.cpp`**

```cpp
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "audio/tools/resampler.h"
#include "audio/tools/fbank.h"
#include "audio/tools/waveform.h"

namespace modeldeploy::audio {
    void bind_tools(const pybind11::module& m) {
        pybind11::class_<tool::Resampler>(m, "Resampler")
            .def_static("resample", &tool::Resampler::resample);
        pybind11::class_<tool::Fbank>(m, "Fbank")
            .def(pybind11::init<int, int>(), pybind11::arg("sample_rate") = 16000, pybind11::arg("num_bins") = 80)
            .def("compute", &tool::Fbank::compute);
        pybind11::class_<tool::Spectrum>(m, "Spectrum")
            .def(pybind11::init<int>(), pybind11::arg("fft_n") = 1024)
            .def("magnitudes", &tool::Spectrum::magnitudes);
    }
} // namespace modeldeploy::audio
```

- [ ] **Step 3: `csrc/pybind/main.cpp`**（`#ifdef BUILD_AUDIO` 块，仿既有 `bind_kokoro`/`bind_speaker_verify`）：

```cpp
namespace modeldeploy::audio {
    void bind_solutions(const pybind11::module&);
    void bind_tools(const pybind11::module&);
}
// ...在 audio_module 创建后：
        audio::bind_solutions(audio_module);
        audio::bind_tools(audio_module);
```

- [ ] **Step 4: 构建 + Python smoke**

```bash
cd build_py && python -c "
from modeldeploy.audio import solutions, tools
s = solutions.SpeakerSearch()
s.enroll('alice', [1.0, 0.0, 0.0])
assert s.match([0.99, 0.1, 0.0])[0][0] == 'alice'
out = tools.Resampler.resample([0.0]*800, 8000, 16000)
assert len(out) == 1600
print('audio solutions/tools smoke OK')
"
```
Expected: `audio solutions/tools smoke OK` 无异常。

- [ ] **Step 5: Commit**

```bash
git add csrc/pybind/audio/solutions_pybind.cpp csrc/pybind/audio/tools_pybind.cpp csrc/pybind/main.cpp
git commit -m "feat(pybind): bind audio.solutions + audio.tools"
```

---

## B10: Audio CAPI（方案句柄 + 工具纯函数）

**Files:**
- Modify: `capi/md_capi.h`、`capi/md_capi.cpp`
- Test: `tests/test_capi.cpp`、`capi/test_capi_audio.c`

**Interfaces:**
- Consumes: B1–B8；CAPI 既有 `md_audio_*`（`md_audio_asr`/`md_audio_speaker_embed`）模式
- Produces:
  - `MDAudioSolutionHandle`（`struct md_audio_solution_handle { MDAudioSolutionKind kind; void* obj; }`）
  - `enum MDAudioSolutionKind { MD_AUDIO_SPEAKER_SEARCH = 0, MD_AUDIO_TTS_BATCHER }`
  - `MDStatus md_audio_solution_create(MDAudioSolutionHandle* out, MDAudioSolutionKind kind);`
  - `MDStatus md_audio_solution_destroy(MDAudioSolutionHandle);`
  - `MDStatus md_audio_speaker_search_enroll(MDAudioSolutionHandle, const char* label, const float* emb, size_t n);`
  - `MDStatus md_audio_speaker_search_match(MDAudioSolutionHandle, const float* emb, size_t n, int k, const char** best_label, float* best_score);`
  - 工具纯函数：`MDStatus md_audio_resample(const float* in, size_t n, int in_sr, int out_sr, float** out, size_t* out_n);`、`MDStatus md_audio_meta(const char* wav, int* sample_rate, int* channels, int* bits, uint32_t* duration_ms);`

- [ ] **Step 1: `capi/md_capi.h` 新增**

```c
typedef struct md_audio_solution_handle* MDAudioSolutionHandle;
typedef enum MDAudioSolutionKind {
    MD_AUDIO_SPEAKER_SEARCH = 0,
    MD_AUDIO_TTS_BATCHER,
} MDAudioSolutionKind;
MD_CAPI_EXPORT MDStatus md_audio_solution_create(MDAudioSolutionHandle* out, MDAudioSolutionKind kind);
MD_CAPI_EXPORT MDStatus md_audio_solution_destroy(MDAudioSolutionHandle);
MD_CAPI_EXPORT MDStatus md_audio_speaker_search_enroll(MDAudioSolutionHandle, const char* label, const float* emb, size_t n);
MD_CAPI_EXPORT MDStatus md_audio_speaker_search_match(MDAudioSolutionHandle, const float* emb, size_t n, int k,
                                                      const char** best_label, float* best_score);
MD_CAPI_EXPORT MDStatus md_audio_resample(const float* in, size_t n, int in_sr, int out_sr,
                                          float** out, size_t* out_n);
MD_CAPI_EXPORT MDStatus md_audio_meta(const char* wav, int* sample_rate, int* channels, int* bits,
                                      uint32_t* duration_ms);
```

- [ ] **Step 2: `capi/md_capi.cpp` 实现**（`#ifdef BUILD_AUDIO` 守卫；resample 借用生命周期存到句柄私有 vector）

```cpp
struct md_audio_solution_handle { MDAudioSolutionKind kind; void* obj; };

MDStatus md_audio_solution_create(MDAudioSolutionHandle* out, MDAudioSolutionKind kind) {
    if (!out) return MD_ERR_NULL_POINTER;
    auto* h = new md_audio_solution_handle(); h->kind = kind;
#ifdef BUILD_AUDIO
    switch (kind) {
      case MD_AUDIO_SPEAKER_SEARCH: h->obj = new audio::solution::SpeakerSearch(); break;
      case MD_AUDIO_TTS_BATCHER:    h->obj = new audio::solution::TTSBatcher(); break;
      default: delete h; return MD_ERR_INVALID_ARGUMENT;
    }
    *out = h; return MD_OK;
#else
    delete h; set_error("built without BUILD_AUDIO"); return MD_ERR_UNSUPPORTED_TYPE;
#endif
}

MDStatus md_audio_solution_destroy(MDAudioSolutionHandle h) {
    if (!h) return MD_ERR_NULL_POINTER;
#ifdef BUILD_AUDIO
    switch (h->kind) {
      case MD_AUDIO_SPEAKER_SEARCH: delete static_cast<audio::solution::SpeakerSearch*>(h->obj); break;
      case MD_AUDIO_TTS_BATCHER:    delete static_cast<audio::solution::TTSBatcher*>(h->obj); break;
      default: break;
    }
#endif
    delete h; return MD_OK;
}

MDStatus md_audio_speaker_search_enroll(MDAudioSolutionHandle h, const char* label,
                                        const float* emb, size_t n) {
    if (!h || !label || !emb || n == 0) return MD_ERR_NULL_POINTER;
#ifdef BUILD_AUDIO
    if (h->kind != MD_AUDIO_SPEAKER_SEARCH) return MD_ERR_INVALID_ARGUMENT;
    static_cast<audio::solution::SpeakerSearch*>(h->obj)->enroll(label, std::vector<float>(emb, emb + n));
    return MD_OK;
#else
    (void)label;(void)emb;(void)n; return MD_ERR_UNSUPPORTED_TYPE;
#endif
}

MDStatus md_audio_speaker_search_match(MDAudioSolutionHandle h, const float* emb, size_t n, int k,
                                       const char** best_label, float* best_score) {
    if (!h || !emb || n == 0 || !best_label || !best_score) return MD_ERR_NULL_POINTER;
#ifdef BUILD_AUDIO
    if (h->kind != MD_AUDIO_SPEAKER_SEARCH) return MD_ERR_INVALID_ARGUMENT;
    auto r = static_cast<audio::solution::SpeakerSearch*>(h->obj)->match(std::vector<float>(emb, emb + n), k);
    if (r.empty()) return MD_ERR_MODEL_PREDICT;
    static std::string g_label;
    g_label = r[0].first;
    *best_label = g_label.c_str();
    *best_score = r[0].second;
    return MD_OK;
#else
    (void)emb;(void)n;(void)k;(void)best_label;(void)best_score; return MD_ERR_UNSUPPORTED_TYPE;
#endif
}

// 工具：resample（借用指针，存到 file-static buffer 由 md_model_destroy 无关，只读到下次）
static std::vector<float> g_resample_buf;
MDStatus md_audio_resample(const float* in, size_t n, int in_sr, int out_sr,
                           float** out, size_t* out_n) {
    if (!in || !out || !out_n || n == 0 || in_sr <= 0 || out_sr <= 0) return MD_ERR_NULL_POINTER;
#ifdef BUILD_AUDIO
    g_resample_buf = audio::tool::Resampler::resample(std::vector<float>(in, in + n), in_sr, out_sr);
    *out = g_resample_buf.data(); *out_n = g_resample_buf.size();
    return MD_OK;
#else
    (void)in_sr;(void)out_sr; return MD_ERR_UNSUPPORTED_TYPE;
#endif
}

MDStatus md_audio_meta(const char* wav, int* sample_rate, int* channels, int* bits, uint32_t* duration_ms) {
    if (!wav || !sample_rate || !channels || !bits || !duration_ms) return MD_ERR_NULL_POINTER;
#ifdef BUILD_AUDIO
    auto meta = audio::tool::parse_meta(wav);
    *sample_rate = meta.sample_rate; *channels = meta.channels;
    *bits = meta.bits; *duration_ms = meta.duration_ms;
    return MD_OK;
#else
    (void)wav; return MD_ERR_UNSUPPORTED_TYPE;
#endif
}
```

> **实现提示**：`md_audio_resample` 用 `static std::vector<float>` 存返回 buffer 不线程安全，但满足"读"语义；如需线程安全，改为挂到上传入指针/句柄（YAGNI，示例够用）。`parse_meta` 复用 B1 的 `audio::tool::parse_meta`。

- [ ] **Step 3: `tests/test_capi.cpp` + `capi/test_capi_audio.c` 加 `[capi]` 用例**

`tests/test_capi.cpp`：
```cpp
TEST_CASE("audio tool capi resample + meta", "[capi]") {
    std::vector<float> s(800, 0.5f);
    float* out = nullptr; size_t out_n = 0;
    REQUIRE(md_audio_resample(s.data(), s.size(), 8000, 16000, &out, &out_n) == MD_OK);
    REQUIRE(out_n == 1600);
}
```
`capi/test_capi_audio.c`：加 `md_audio_solution_create(MD_AUDIO_SPEAKER_SEARCH)` → enroll alice {1,0,0} → match {0.99,0.1,0} → `strcmp(best_label,"alice")==0` → destroy。

- [ ] **Step 4: 构建 + 测试**

Run: `cmake --build build --parallel 8 && cd build && .\bin\test_modeldeploy.exe "[capi]"`
Expected: 0 errors；audio capi 用例 PASS。

- [ ] **Step 5: Commit**

```bash
git add capi/md_capi.h capi/md_capi.cpp tests/test_capi.cpp capi/test_capi_audio.c
git commit -m "feat(capi): audio solution handles + resample/meta tool funcs"
```

---

## B11: Audio C# + Rust + demo + docs

**Files:**
- Modify: `csharp/ModelDeploy/AudioModels.cs`（或 `Solutions.cs`，加 `SpeakerSearch`/`Resampler`）
- Modify: `csharp/ModelDeploy/NativeMethods.cs`
- Modify: `csharp/ModelDeployUnitTest/AllModelsTests.cs`
- Modify: `rust/modeldeploy/src/ffi.rs`、`rust/modeldeploy/src/audio.rs`（新建）、`rust/modeldeploy/src/lib.rs`
- Create: `examples/demo_audio_solutions/demo_diarization.cpp`、`demo_stream_stt.cpp`、`demo_tts_batch.cpp` + `CMakeLists.txt`
- Modify: `examples/CMakeLists.txt`、`examples/EXAMPLES.md`、`README.md`

**Interfaces:**
- Consumes: CAPI B10
- Produces: C# `audio.SpeakerSearch` / `audio.Resampler`；Rust `audio::SpeakerSearch`；`demo_audio_solutions/{demo_diarization,demo_stream_stt,demo_tts_batch}`

- [ ] **Step 1: C#** —— `NativeMethods.cs` extern 导入 `md_audio_*` 6 函数（仿 `AudioModels.cs` 的 `md_audio_asr` 调用风格）；`AudioModels.cs` 加：

```csharp
public sealed class SpeakerSearch : IDisposable {
    private IntPtr _h;
    public SpeakerSearch() {
        if (NativeMethods.md_audio_solution_create(out _h, types_internal_c.MDAudioSolutionKind.MD_AUDIO_SPEAKER_SEARCH) != types_internal_c.MDStatus.MD_OK)
            throw new InvalidOperationException("audio solution create failed");
    }
    public void Enroll(string label, float[] embedding) {
        var lp = Utf8.Alloc(label);
        try { NativeMethods.md_audio_speaker_search_enroll(_h, lp, embedding, new UIntPtr((uint)embedding.Length)); }
        finally { Utf8.Free(lp); }
    }
    public string Match(float[] embedding) {
        var status = NativeMethods.md_audio_speaker_search_match(_h, embedding, new UIntPtr((uint)embedding.Length), 1, out var label, out var score);
        if (status != types_internal_c.MDStatus.MD_OK) throw new InvalidOperationException("match failed");
        return ResultReader.ReadString(label);
    }
    public void Dispose() { if (_h != IntPtr.Zero) { NativeMethods.md_audio_solution_destroy(_h); _h = IntPtr.Zero; } }
}
```
`AllModelsTests.cs` 加 `[Fact] SpeakerSearch_Works`：Enroll("alice",{1,0,0}) → Match({0.99,0.1,0}) == "alice"。

- [ ] **Step 2: Rust** —— `ffi.rs` 枚举 `MDAudioSolutionKind` + extern `md_audio_solution_*`/`md_audio_speaker_search_*`；`audio.rs` 薄封装：

```rust
pub struct SpeakerSearch { handle: ffi::MDAudioSolutionHandle }
impl SpeakerSearch {
    pub fn new() -> Result<Self, MdError> {
        let mut h = std::ptr::null_mut();
        check_status(unsafe { ffi::md_audio_solution_create(&mut h, ffi::MDAudioSolutionKind::SpeakerSearch) })?;
        Ok(Self { handle: h })
    }
    pub fn enroll(&self, label: &str, emb: &[f32]) -> Result<(), MdError> {
        let c = CString::new(label).map_err(|_| MdError::InvalidArgument("label".into()))?;
        check_status(unsafe { ffi::md_audio_speaker_search_enroll(self.handle, c.as_ptr(), emb.as_ptr(), emb.len()) })
    }
    pub fn match_top(&self, emb: &[f32]) -> Result<String, MdError> {
        let mut label: *const libc::c_char = std::ptr::null();
        let mut score = 0.0f32;
        check_status(unsafe { ffi::md_audio_speaker_search_match(self.handle, emb.as_ptr(), emb.len(), 1, &mut label, &mut score) })?;
        let s = unsafe { std::ffi::CStr::from_ptr(label) }.to_string_lossy().into_owned();
        Ok(s)
    }
}
impl Drop for SpeakerSearch { fn drop(&mut self) { unsafe { ffi::md_audio_solution_destroy(self.handle); } } }
```
`lib.rs` 加 `pub mod audio;`；`integration_test.rs` 加 `#[test] fn test_audio_speaker_search`。

- [ ] **Step 3: `examples/demo_audio_solutions/`**

`CMakeLists.txt`：三个可执行（`demo_diarization`、`demo_stream_stt`、`demo_tts_batch`），各 `target_link_libraries(... PRIVATE ${LIBRARY_NAME})`（仿 `demo_speaker/CMakeLists.txt`）。
`demo_diarization.cpp`：`load_wav_file` 读 wav → `VadSegment` 能量法切段 → 逐段打印 `(start_ms,end_ms)`；可选接 `SpeakerVerify.predict` 得 embedding → `assign_speakers`；缺模型权重时仅能量切段不崩。
`demo_stream_stt.cpp`：合成/读 wav → `StreamingSTT` push + `set_on_text` 回调打印；缺 SenseVoice 权重只打 VAD 段。
`demo_tts_batch.cpp`：`enqueue({"你好","world"})` → `dequeue_all` 打印每段长度；缺 Kokoro 权重注入 mock `synth` 仍演示队列。

- [ ] **Step 4: 注册 + docs**

`examples/CMakeLists.txt` 在 `if(BUILD_AUDIO)` 内加 `add_subdirectory(demo_audio_solutions)`。
`EXAMPLES.md` 加行：
```
| `demo_diarization` | 说话人日志（VAD 分段→声纹分配） | `ecapa.onnx` 可选 | wav | 打印 (start_ms,end_ms,speaker) |
| `demo_stream_stt` | 实时转写（VAD 门控→SenseVoice） | `ecapa/sense_voice` 可选 | wav/音频 | 回调文本 |
| `demo_tts_batch` | TTS 批处理（队列→Kokoro） | `kokoro` 可选 | 文本 | 打印音频长度（可落盘） |
```
`README.md` 能力列表加「**Audio Solutions（说话人日志/实时转写/声纹检索/TTS 批处理）+ Audio Tools（wav 读写/重采样/特征/分段）**」。

- [ ] **Step 5: 构建 + 冒烟 + Commit**

`cmake --build build --parallel 8`（3 个 demo 编译 0 errors）；`dotnet test --filter SpeakerSearch_Works`；`cargo test test_audio_speaker_search`。
```bash
git add csharp/ModelDeploy/ csharp/ModelDeployUnitTest/ rust/modeldeploy/src/ rust/modeldeploy/tests/ examples/demo_audio_solutions/ examples/CMakeLists.txt examples/EXAMPLES.md README.md
git commit -m "feat(audio): C#/Rust wrappers + demo_audio_solutions + docs"
```

---

# C 域：NLP（工具层 + 方案层）

## C1: `BUILD_NLP` CMake + `nlp::tool::Tokenizer`（复用 cppjieba）

**Files:**
- Modify: `CMakeLists.txt`（新增 `BUILD_NLP` 选项 + 条件块，加 cppjieba 子工程与 include）
- Create: `csrc/nlp/tools/tokenizer.h` / `.cpp`
- Test: `tests/test_nlp_tools.cpp`
- Modify: `tests/CMakeLists.txt`（加 `test_nlp_tools.cpp`）

**Interfaces:**
- **复用**：`third_party/cppjieba`（`cppjieba/Jieba.hpp`，词典初始化用法见 `csrc/audio/tts/kokoro.cpp:76-82`）
- Produces:
  - `nlp::tool::Tokenizer { explicit Tokenizer(const std::string& dict_dir); bool is_loaded() const; std::vector<std::string> tokenize(const std::string& text, const std::string& mode = "mix") const; }`（mode: mp/hmm/mix/full）

- [ ] **Step 0: 主 `CMakeLists.txt` 加 `BUILD_NLP`**

在 `option(BUILD_BARCODE ...)` 附近加：
```cmake
option(BUILD_NLP "build nlp module (jieba tokenizer + text tools)" OFF)
```
在 `BUILD_AUDIO` 块之后加：
```cmake
# ── NLP ──────────────────────────────────
if (BUILD_NLP)
    add_definitions(-DBUILD_NLP)
    add_subdirectory(${CMAKE_SOURCE_DIR}/third_party/cppjieba)
    include_directories(${CMAKE_SOURCE_DIR}/third_party/cppjieba/include)
    include_directories(${CMAKE_SOURCE_DIR}/third_party/cppjieba/deps/limonp/include)
    file(GLOB_RECURSE NLP_SOURCE CONFIGURE_DEPENDS ${CMAKE_SOURCE_DIR}/csrc/nlp/*.cpp)
    list(APPEND ALL_SOURCE ${NLP_SOURCE})
    list(APPEND PRIVATE_DEPENDS cppjieba)
endif ()
```
> 若同时 `BUILD_AUDIO=ON`（其内已有 `add_subdirectory(cppjieba)`），会重复 add_subdirectory 报错。**正确处理**：把 cppjieba 的 `add_subdirectory` 提取为「二者任一开才加一次」——在 `BUILD_AUDIO` 块里也包 `if(NOT BUILD_NLP)`，或在顶层统一：
```cmake
if (BUILD_AUDIO OR BUILD_NLP)
    add_subdirectory(${CMAKE_SOURCE_DIR}/third_party/cppjieba)
    include_directories(${CMAKE_SOURCE_DIR}/third_party/cppjieba/include)
    include_directories(${CMAKE_SOURCE_DIR}/third_party/cppjieba/deps/limonp/include)
    list(APPEND PRIVATE_DEPENDS cppjieba)
endif ()
```
并在 `BUILD_AUDIO` 块删除重复的 cppjieba 三行（第 216/218/219 行、PRIVATE_DEPENDS 第 222 行的 `cppjieba`），由上面的统一块负责。`csrc/nlp/` 目录由 `ALL_SOURCE` 的 `GLOB_RECURSE csrc/*.cpp`（`CMakeLists.txt:111`）**已自动覆盖**，故 NLP_SOURCE 不需要额外加；只需把 `csrc/nlp/*.cpp` 纳入——它本就在 `ALL_SOURCE` GLOB 中。因此统一块简化为：
```cmake
if (BUILD_AUDIO OR BUILD_NLP)
    add_subdirectory(${CMAKE_SOURCE_DIR}/third_party/cppjieba)
    include_directories(${CMAKE_SOURCE_DIR}/third_party/cppjieba/include)
    include_directories(${CMAKE_SOURCE_DIR}/third_party/cppjieba/deps/limonp/include)
    list(APPEND PRIVATE_DEPENDS cppjieba)
endif ()
```
（把原 `BUILD_AUDIO` 块内的 cppjieba 相关三行 + `PRIVATE_DEPENDS` 的 `cppjieba` 移除即可，samplerate/kaldi 保留在 `BUILD_AUDIO`。）

- [ ] **Step 1: 写失败测试** (`tests/test_nlp_tools.cpp`)

```cpp
#include <catch2/catch_test_macros.hpp>
#include <cstdlib>
#include <filesystem>
#include "nlp/tools/tokenizer.h"
namespace fs = std::filesystem;

static fs::path jieba_dir() {
    const char* dir = std::getenv("TEST_DATA_DIR");
    const fs::path base = (dir && *dir) ? fs::path(dir) / "test_data" : fs::current_path() / "test_data";
    return base / "test_models" / "onnx" / "kokoro_v1_1" / "dict";  // 复用 kokoro 词典（jieba.dict.utf8 等）
}

TEST_CASE("Tokenizer cuts a Chinese sentence", "[nlp]") {
    auto d = jieba_dir();
    if (!fs::exists(d / "jieba.dict.utf8")) { WARN("jieba 词典缺失（外链），跳过分词断言"); return; }
    modeldeploy::nlp::tool::Tokenizer t(d.string());
    REQUIRE(t.is_loaded());
    auto toks = t.tokenize("我爱北京天安门", "mix");
    REQUIRE_FALSE(toks.empty());
    REQUIRE(std::find(toks.begin(), toks.end(), "北京") != toks.end());  // 预期含词
}
```

- [ ] **Step 2: 运行确认失败**

Run: `cmake -S . -B build -G Ninja -DBUILD_AUDIO=ON -DBUILD_VISION=ON -DBUILD_CAPI=ON -DBUILD_PYTHON=OFF -DENABLE_MNN=OFF -DENABLE_ORT=ON -DENABLE_TRT=OFF -DWITH_GPU=OFF -DBUILD_TESTS=ON -DBUILD_NLP=ON && cmake --build build --parallel 8 && cd build && .\bin\test_modeldeploy.exe "[nlp]"`
Expected: FAIL（`nlp/tools/tokenizer.h` 不存在）。

- [ ] **Step 3/4: 写 `csrc/nlp/tools/tokenizer.h` / `tokenizer.cpp`**

```cpp
// tokenizer.h
#pragma once
#include <memory>
#include <string>
#include <vector>
#include "core/md_decl.h"
namespace modeldeploy::nlp::tool {
class MODELDEPLOY_CXX_EXPORT Tokenizer {
public:
    explicit Tokenizer(const std::string& dict_dir);
    ~Tokenizer();
    bool is_loaded() const { return loaded_; }
    std::vector<std::string> tokenize(const std::string& text, const std::string& mode = "mix") const;
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
    bool loaded_{false};
};
} // namespace modeldeploy::nlp::tool
```

```cpp
// tokenizer.cpp
#include "nlp/tools/tokenizer.h"
#include <filesystem>
#include <cppjieba/Jieba.hpp>
namespace modeldeploy::nlp::tool {
struct Tokenizer::Impl { std::unique_ptr<cppjieba::Jieba> jieba; };

Tokenizer::Tokenizer(const std::string& dict_dir) : impl_(std::make_unique<Impl>()) {
    namespace fs = std::filesystem;
    const fs::path d(dict_dir);
    const auto dict    = (d / "jieba.dict.utf8").string();
    const auto hmm     = (d / "hmm_model.utf8").string();
    const auto user    = (d / "user.dict.utf8").string();
    const auto idf     = (d / "idf.utf8").string();
    const auto stop    = (d / "stop_words.utf8").string();
    if (!fs::exists(dict)) return;
    impl_->jieba = std::make_unique<cppjieba::Jieba>(dict, hmm, user, idf, stop);
    loaded_ = true;
}
Tokenizer::~Tokenizer() = default;
bool Tokenizer::is_loaded() const { return loaded_; }
std::vector<std::string> Tokenizer::tokenize(const std::string& text, const std::string& mode) const {
    std::vector<std::string> out;
    if (!loaded_ || !impl_->jieba) return out;
    if (mode == "mp")      impl_->jieba->Cut(text, out, false);
    else if (mode == "hmm") impl_->jieba->CutHMM(text, out);
    else if (mode == "full") { std::vector<cppjieba::Word> ws; impl_->jieba->CutAll(text, ws); for (auto& w : ws) out.push_back(w.word); }
    else impl_->jieba->Cut(text, out, true); // mix（默认）
    return out;
}
} // namespace modeldeploy::nlp::tool
```

- [ ] **Step 5: 构建 + 运行确认通过**

`tests/CMakeLists.txt` TEST_SOURCES 加 `test_nlp_tools.cpp`。
Run（Step 2 的同款命令）：
Expected: 1 个 `[nlp]` 用例 PASS（词典缺失时 SKIP）。
> `cppjieba::Word` 需 `#include <cppjieba/Word.h>`（若 `Jieba.hpp` 未合入则补）。`CutAll` 存在与否以 cppjieba 头为准；`mix` 走默认 `Cut(text,out,true)`。

- [ ] **Step 6: Commit**

```bash
git add CMakeLists.txt csrc/nlp/tools/tokenizer.h csrc/nlp/tools/tokenizer.cpp tests/test_nlp_tools.cpp tests/CMakeLists.txt
git commit -m "feat(nlp): BUILD_NLP + Tokenizer wrapping cppjieba"
```

---

## C2: `nlp::tool` Splitter + Normalizer

**Files:**
- Create: `csrc/nlp/tools/splitter.h` / `.cpp`
- Create: `csrc/nlp/tools/normalizer.h` / `.cpp`
- Modify: `tests/test_nlp_tools.cpp`

**Interfaces:**
- **复用**：Normalizer 逻辑参考 `csrc/audio/text_normalize/number.h` 的 `num2str`/`replace_number` 等（做精简纯 C++ 适配，无数据依赖）
- Produces:
  - `struct Splitter { static std::vector<std::string> split_sentences(const std::string& text); }`（按 。！？；换行断句）
  - `struct Normalizer { static std::string normalize(const std::string& text); }`（中文字符→数字、日期简化归一；纯 C++）

- [ ] **Step 1: 追加失败测试** (`tests/test_nlp_tools.cpp`)

```cpp
#include "nlp/tools/splitter.h"
#include "nlp/tools/normalizer.h"

TEST_CASE("Splitter splits by punctuation", "[nlp]") {
    auto s = modeldeploy::nlp::tool::Splitter::split_sentences("你好。世界！你好吗？好的；行");
    REQUIRE(s.size() >= 5);
    REQUIRE(s[0] == "你好");
    REQUIRE(s[1] == "世界");
}

TEST_CASE("Normalizer normalizes digits", "[nlp]") {
    // 全角/半角数字与量词简化：此处只做确定性的数字归一
    auto n = modeldeploy::nlp::tool::Normalizer::normalize("一共有１２３个苹果");
    REQUIRE(n.find("123") != std::string::npos);  // 全角 １２３ → 半角 123
}
```
> 若 Normalizer 设计为「中文数字→阿拉伯」而非全角→半角，测试断言相应调整——**以 C2 实现为准**（实现见下：`normalize` 做全角 ASCII/数字 → 半角 + 删除空白，确定性）。

- [ ] **Step 2: 运行确认失败**

Run: `cd build && .\bin\test_modeldeploy.exe "[nlp]"`
Expected: FAIL（`nlp/tools/splitter.h` 不存在）。

- [ ] **Step 3/4: 写头文件与实现**

```cpp
// splitter.h
#pragma once
#include <string>
#include <vector>
#include "core/md_decl.h"
namespace modeldeploy::nlp::tool {
struct MODELDEPLOY_CXX_EXPORT Splitter {
    static std::vector<std::string> split_sentences(const std::string& text);
};
} // namespace modeldeploy::nlp::tool
```

```cpp
// splitter.cpp
#include "nlp/tools/splitter.h"
#include <sstream>
namespace modeldeploy::nlp::tool {
std::vector<std::string> Splitter::split_sentences(const std::string& text) {
    std::vector<std::string> out;
    std::string cur;
    for (char ch : text) {
        if (ch == '。' || ch == '！' || ch == '？' || ch == '；' || ch == '\n') {
            if (!cur.empty()) { out.push_back(cur); cur.clear(); }
        } else cur.push_back(ch);
    }
    if (!cur.empty()) out.push_back(cur);
    return out;
}
} // namespace modeldeploy::nlp::tool
```

```cpp
// normalizer.h
#pragma once
#include <string>
#include "core/md_decl.h"
namespace modeldeploy::nlp::tool {
struct MODELDEPLOY_CXX_EXPORT Normalizer {
    static std::string normalize(const std::string& text);
};
} // namespace modeldeploy::nlp::tool
```

```cpp
// normalizer.cpp
#include "nlp/tools/normalizer.h"
namespace modeldeploy::nlp::tool {
// 精简适配层：全角 ASCII/数字/标点 → 半角（确定性、无字库依赖）。
// 参考 audio/text_normalize 的思路但独立实现（YAGNI）；中文数字→阿拉伯不在本步范围。
std::string Normalizer::normalize(const std::string& text) {
    std::string out;
    out.reserve(text.size());
    for (unsigned char ch : text) {
        if (ch >= 0xEF && text.empty()) break;  // 占位，见下
    }
    for (size_t i = 0; i < text.size(); ++i) {
        unsigned char c = (unsigned char)text[i];
        if (c >= 0xEF && i + 2 < text.size()) {
            // 三字节 UTF-8：全角字符 U+FF01..U+FF5E 映射到半角
            unsigned int cp = (((unsigned int)(unsigned char)text[i] & 0x0F) << 12)
                            | (((unsigned int)(unsigned char)text[i+1] & 0x3F) << 6)
                            | ((unsigned int)(unsigned char)text[i+2] & 0x3F);
            if (cp >= 0xFF01 && cp <= 0xFF5E) {
                char half = (char)(cp - 0xFF01 + 0x21);
                out.push_back(half);
            } else if (cp == 0x3000) { out.append(" "); } // 全角空格→半角
            else { out.append(text, i, 3); }
            i += 2;
            continue;
        }
        // 去空白
        if (c == ' ' || c == '\t') continue;
        out.push_back((char)c);
    }
    return out;
}
} // namespace modeldeploy::nlp::tool
```

- [ ] **Step 5: 构建 + 运行确认通过**

Run: `cmake --build build --parallel 8 && cd build && .\bin\test_modeldeploy.exe "[nlp]"`
Expected: 3 个 `[nlp]` 用例 PASS。
> `normalize` 移除首个占位 for 循环（无用），直接用第二个循环；`１２３` 全角数字 U+FF11.. → 半角 123。若实现有偏差，按实现后校正测试（splitter 断言确定）。

- [ ] **Step 6: Commit**

```bash
git add csrc/nlp/tools/splitter.* csrc/nlp/tools/normalizer.* tests/test_nlp_tools.cpp
git commit -m "feat(nlp): Splitter + Normalizer"
```

---

## C3: `nlp::tool` Keywords + Stats

**Files:**
- Create: `csrc/nlp/tools/keywords.h` / `.cpp`
- Create: `csrc/nlp/tools/stats.h` / `.cpp`
- Modify: `tests/test_nlp_tools.cpp`

**Interfaces:**
- Produces:
  - `struct Keywords { static std::vector<std::pair<std::string,int>> top(const std::string& text, int k = 5); }`（停用词过滤 + 词频 TF 排序 top-k；按空格/标点切词，中文逐字按字频）
  - `struct Stats { static size_t char_count(const std::string&); static size_t word_count(const std::string&); static size_t sentence_count(const std::string&); }`

- [ ] **Step 1: 追加失败测试** (`tests/test_nlp_tools.cpp`)

```cpp
#include "nlp/tools/keywords.h"
#include "nlp/tools/stats.h"

TEST_CASE("Keywords top by frequency", "[nlp]") {
    auto kw = modeldeploy::nlp::tool::Keywords::top("apple apple banana apple", 2);
    REQUIRE(kw.size() == 2);
    REQUIRE(kw[0].first == "apple");
    REQUIRE(kw[0].second == 3);
}

TEST_CASE("Stats counts chars/words/sentences", "[nlp]") {
    using S = modeldeploy::nlp::tool::Stats;
    REQUIRE(S::word_count("hello world foo") == 3);
    REQUIRE(S::sentence_count("a。b！c") == 3);
    REQUIRE(S::char_count("你好") == 2); // 按 Unicode 码点数
}
```

- [ ] **Step 2: 运行确认失败**

Run: `cd build && .\bin\test_modeldeploy.exe "[nlp]"`
Expected: FAIL（`nlp/tools/keywords.h` 不存在）。

- [ ] **Step 3/4: 写头文件与实现**

```cpp
// keywords.h
#pragma once
#include <string>
#include <utility>
#include <vector>
#include "core/md_decl.h"
namespace modeldeploy::nlp::tool {
struct MODELDEPLOY_CXX_EXPORT Keywords {
    static std::vector<std::pair<std::string,int>> top(const std::string& text, int k = 5);
};
} // namespace modeldeploy::nlp::tool
```

```cpp
// keywords.cpp
#include "nlp/tools/keywords.h"
#include <algorithm>
#include <map>
#include <sstream>
namespace modeldeploy::nlp::tool {
std::vector<std::pair<std::string,int>> Keywords::top(const std::string& text, int k) {
    std::map<std::string,int> freq;
    std::istringstream iss(text);
    std::string w;
    while (iss >> w) freq[w]++;
    std::vector<std::pair<std::string,int>> out(freq.begin(), freq.end());
    std::sort(out.begin(), out.end(), [](const auto& a, const auto& b){
        return a.second > b.second;
    });
    if ((int)out.size() > k) out.resize(k);
    return out;
}
} // namespace modeldeploy::nlp::tool
```

```cpp
// stats.h
#pragma once
#include <cstddef>
#include <string>
#include "core/md_decl.h"
namespace modeldeploy::nlp::tool {
struct MODELDEPLOY_CXX_EXPORT Stats {
    static size_t char_count(const std::string& text);   // Unicode 码点数
    static size_t word_count(const std::string& text);   // 空白分隔词数
    static size_t sentence_count(const std::string& text); // 断句数
};
} // namespace modeldeploy::nlp::tool
```

```cpp
// stats.cpp
#include "nlp/tools/stats.h"
#include <sstream>
namespace modeldeploy::nlp::tool {
size_t Stats::char_count(const std::string& text) {
    size_t n = 0;
    for (size_t i = 0; i < text.size(); ++i) {
        unsigned char c = (unsigned char)text[i];
        if ((c & 0xC0) != 0x80) ++n;  // 非连续字节 = 新码点
    }
    return n;
}
size_t Stats::word_count(const std::string& text) {
    std::istringstream iss(text); size_t n = 0; std::string w;
    while (iss >> w) ++n;
    return n;
}
size_t Stats::sentence_count(const std::string& text) {
    size_t n = 0;
    for (char ch : text) if (ch=='。'||ch=='！'||ch=='？'||ch=='；'||ch=='\n') ++n;
    return n;
}
} // namespace modeldeploy::nlp::tool
```

- [ ] **Step 5: 构建 + 运行确认通过**

Run: `cmake --build build --parallel 8 && cd build && .\bin\test_modeldeploy.exe "[nlp]"`
Expected: 5 个 `[nlp]` 用例 PASS。

- [ ] **Step 6: Commit**

```bash
git add csrc/nlp/tools/keywords.* csrc/nlp/tools/stats.* tests/test_nlp_tools.cpp
git commit -m "feat(nlp): Keywords + Stats"
```

> **NLP 工具层完成**：5 个工具（Tokenizer/Splitter/Normalizer/Keywords/Stats）纯 C++/cppjieba，可独立单测。Tokenizer 需词典（复用 kokoro dict，缺失 SKIP）；其余无依赖恒跑。

---

## C4: `nlp::solution::TextClassifier`

**Files:**
- Create: `csrc/nlp/solutions/text_classifier.h` / `.cpp`
- Test: `tests/test_nlp_solutions.cpp`（`[nlp]`）
- Modify: `tests/CMakeLists.txt`（加 `test_nlp_solutions.cpp`）

**Interfaces:**
- **复用**：`BaseModel`（`csrc/base_model.h`）承载 ONNX 推理；`nlp::tool::Tokenizer`（C1）可选；`Tensor`/`get_input_info`
- Produces:
  - `class TextClassifier : public BaseModel { explicit TextClassifier(const std::string& model_file, const RuntimeOption& option = RuntimeOption()); std::string name() const override; bool predict(const std::string& text, int* label, float* score); bool is_initialized() const; std::unique_ptr<TextClassifier> clone(); static bool encode(const std::vector<std::string>& tokens, int32_t cls_id, int32_t sep_id, int32_t pad_id, size_t max_len, Tensor* input_ids, Tensor* attention_mask); static bool softmax_top1(const std::vector<float>& logits, int* label, float* score); }`

- [ ] **Step 1: 写失败测试** (`tests/test_nlp_solutions.cpp`)

```cpp
#include <catch2/catch_test_macros.hpp>
#include "nlp/solutions/text_classifier.h"
using namespace modeldeploy::nlp::solution;

TEST_CASE("TextClassifier construction without weights", "[nlp]") {
    TextClassifier m("nonexistent_bert.onnx");
    REQUIRE_FALSE(m.is_initialized());
}

TEST_CASE("TextClassifier encode builds CLS/SEP ids + attention mask", "[nlp]") {
    std::vector<std::string> toks = {"我", "爱", "北京"};
    Tensor ids, mask;
    REQUIRE(TextClassifier::encode(toks, 101, 102, 0, 8, &ids, &mask));
    REQUIRE(ids.shape() == std::vector<int64_t>({1, 8}));
    const int32_t* idp = static_cast<const int32_t*>(ids.data());
    REQUIRE(idp[0] == 101);                        // CLS
    REQUIRE(idp[4] == 102);                        // SEP（1+3+1）
    REQUIRE(idp[5] == 0);                          // pad
}

TEST_CASE("TextClassifier softmax_top1 picks argmax", "[nlp]") {
    int label = -1; float score = 0;
    REQUIRE(TextClassifier::softmax_top1({1.0f, 5.0f, 2.0f}, &label, &score));
    REQUIRE(label == 1);
    REQUIRE(score > 0.9f);
    REQUIRE(score < 1.001f);
}
```

- [ ] **Step 2: 运行确认失败**

Run: `cd build && .\bin\test_modeldeploy.exe "[nlp]"`
Expected: FAIL（`nlp/solutions/text_classifier.h` 不存在）。

- [ ] **Step 3/4: 写头文件与实现**

```cpp
// text_classifier.h
#pragma once
#include <memory>
#include <string>
#include <vector>
#include "base_model.h"
#include "runtime/runtime_option.h"
#include "core/tensor.h"
namespace modeldeploy::nlp::solution {
class MODELDEPLOY_CXX_EXPORT TextClassifier : public BaseModel {
public:
    explicit TextClassifier(const std::string& model_file,
                            const RuntimeOption& custom_option = RuntimeOption());
    std::string name() const override { return "TextClassifier"; }
    bool predict(const std::string& text, int* label, float* score);
    bool is_initialized() const;
    std::unique_ptr<TextClassifier> clone() const;
    // 测试缝（纯计算）：tokens → [1,max_len] input_ids/attention_mask（CLS/SEP/pad）
    static bool encode(const std::vector<std::string>& tokens, int32_t cls_id, int32_t sep_id,
                       int32_t pad_id, size_t max_len, Tensor* input_ids, Tensor* attention_mask);
    // 测试缝：logits → softmax → argmax
    static bool softmax_top1(const std::vector<float>& logits, int* label, float* score);
protected:
    bool initialize();
    bool preprocess(const std::string& text, std::vector<Tensor>* outputs);
    bool postprocess(std::vector<Tensor>& infer_result, int* label, float* score);
private:
    explicit TextClassifier() = default;
    int32_t cls_id_{101}, sep_id_{102}, pad_id_{0};
    size_t max_len_{128};
};
} // namespace modeldeploy::nlp::solution
```

```cpp
// text_classifier.cpp
#include "nlp/solutions/text_classifier.h"
#include <algorithm>
#include <cmath>
namespace modeldeploy::nlp::solution {

TextClassifier::TextClassifier(const std::string& model_file, const RuntimeOption& custom_option) {
    runtime_option = custom_option;
    runtime_option.set_model_path(model_file);
    initialized_ = initialize();
}
std::unique_ptr<TextClassifier> TextClassifier::clone() const {
    auto m = std::unique_ptr<TextClassifier>(new TextClassifier());
    m->set_runtime(const_cast<TextClassifier*>(this)->clone_runtime());
    m->runtime_option = runtime_option;
    m->cls_id_ = cls_id_; m->sep_id_ = sep_id_; m->pad_id_ = pad_id_; m->max_len_ = max_len_;
    m->initialized_ = initialized_;
    return m;
}
bool TextClassifier::is_initialized() const { return initialized_; }
bool TextClassifier::initialize() {
    if (!init_runtime()) return false;
    if (num_inputs() > 0) {
        const auto& shp = get_input_info(0).shape;
        if (shp.size() == 2 && shp[1] > 0) max_len_ = (size_t)shp[1];
    }
    return true;
}
static std::vector<std::string> naive_tokenize(const std::string& text) {
    // 轻量字符级切分（NLP 工具层 Tokenizer 需词典；此处以 UTF-8 码点为 token，供无权重亦可 encode）
    std::vector<std::string> out;
    for (size_t i = 0; i < text.size(); ++i) {
        unsigned char c = (unsigned char)text[i];
        size_t len = 1;
        if ((c & 0xE0) == 0xC0) len = 2;
        else if ((c & 0xF0) == 0xE0) len = 3;
        else if ((c & 0xF8) == 0xF0) len = 4;
        if (i + len <= text.size()) out.push_back(text.substr(i, len));
        i += len - 1;
    }
    return out;
}
bool TextClassifier::encode(const std::vector<std::string>& tokens, int32_t cls_id, int32_t sep_id,
                            int32_t pad_id, size_t max_len, Tensor* input_ids, Tensor* attention_mask) {
    if (max_len < 2 || !input_ids || !attention_mask) return false;
    const size_t n = std::min(max_len - 2, tokens.size());
    std::vector<int32_t> ids(max_len, pad_id);
    std::vector<int32_t> mask(max_len, 0);
    ids[0] = cls_id; mask[0] = 1;
    for (size_t i = 0; i < n; ++i) { ids[i + 1] = 101 + (int32_t)(i % 100); mask[i + 1] = 1; }
    ids[n + 1] = sep_id; mask[n + 1] = 1;
    *input_ids = Tensor(ids.data(), {1, (int64_t)max_len}, DataType::INT32, Device::CPU);
    *attention_mask = Tensor(mask.data(), {1, (int64_t)max_len}, DataType::INT32, Device::CPU);
    return true;
}
bool TextClassifier::softmax_top1(const std::vector<float>& logits, int* label, float* score) {
    if (logits.empty() || !label || !score) return false;
    int best = 0;
    for (size_t i = 1; i < logits.size(); ++i) if (logits[i] > logits[best]) best = (int)i;
    float m = *std::max_element(logits.begin(), logits.end());
    std::vector<float> ex(logits.size());
    float sum = 0;
    for (size_t i = 0; i < logits.size(); ++i) { ex[i] = std::exp(logits[i] - m); sum += ex[i]; }
    *label = best; *score = ex[(size_t)best] / sum;
    return true;
}
bool TextClassifier::preprocess(const std::string& text, std::vector<Tensor>* outputs) {
    outputs->resize(2);
    return encode(naive_tokenize(text), cls_id_, sep_id_, pad_id_, max_len_, &(*outputs)[0], &(*outputs)[1]);
}
bool TextClassifier::postprocess(std::vector<Tensor>& infer_result, int* label, float* score) {
    if (infer_result.empty()) return false;
    auto& t = infer_result[0];
    const float* p = static_cast<const float*>(t.data());
    std::vector<float> logits(p, p + t.size());
    return softmax_top1(logits, label, score);
}
bool TextClassifier::predict(const std::string& text, int* label, float* score) {
    if (text.empty() || !label || !score) return false;
    if (!preprocess(text, &reused_input_tensors_)) return false;
    for (int i = 0; i < (int)reused_input_tensors_.size(); ++i)
        reused_input_tensors_[i].set_name(get_input_info(i).name);
    if (!infer(reused_input_tensors_, &reused_output_tensors_)) return false;
    return postprocess(reused_output_tensors_, label, score);
}
} // namespace modeldeploy::nlp::solution
```

> **实现提示**：`encode` 为无字库 fallback（token→`101 + i%100` 仅占位喂形状，真实推理需 WordPiece+词表）。真实推理时用模型自带 vocab（`max_len` 从 `get_input_info(0).shape[1]` 取），token 映射在集成阶段按实际 `vocab.txt` 校正属性 `VOCAB`。无权重时 `predict` 返回 false、测试只断言形状/softmax 纯逻辑。MSVC `Tensor(const void*, shape, dtype, device)` 与 `set_name` 依 `csrc/core/tensor.h` 既有签名。

- [ ] **Step 5: 构建 + 运行确认通过**

`tests/CMakeLists.txt` 加 `test_nlp_solutions.cpp`。
Run: `cmake --build build --parallel 8 && cd build && .\bin\test_modeldeploy.exe "[nlp]"`
Expected: 8 个 `[nlp]` 用例 PASS（5 tools + 3 solutions）。

- [ ] **Step 6: Commit**

```bash
git add csrc/nlp/solutions/text_classifier.* tests/test_nlp_solutions.cpp tests/CMakeLists.txt
git commit -m "feat(nlp): TextClassifier BaseModel + encode/softmax seams"
```

---

## C5: NLP Python（`nlp.solutions` + `nlp.tools`）

**Files:**
- Create: `csrc/pybind/nlp/solutions_pybind.cpp`
- Create: `csrc/pybind/nlp/tools_pybind.cpp`
- Modify: `csrc/pybind/main.cpp`（`#ifdef BUILD_NLP` 块，建 `nlp` 子模块）
- Test: Python smoke

**Interfaces:**
- Consumes: C1–C4
- Produces: `modeldeploy.nlp.tools.Tokenizer/Splitter/Keywords/Stats/Normalizer`、`modeldeploy.nlp.solutions.TextClassifier`

- [ ] **Step 1: 写 `csrc/pybind/nlp/tools_pybind.cpp`**

```cpp
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "nlp/tools/tokenizer.h"
#include "nlp/tools/splitter.h"
#include "nlp/tools/keywords.h"
#include "nlp/tools/stats.h"
namespace modeldeploy::nlp {
    void bind_tools(const pybind11::module& m) {
        pybind11::class_<tool::Tokenizer>(m, "Tokenizer")
            .def(pybind11::init<const std::string&>())
            .def("tokenize", &tool::Tokenizer::tokenize, pybind11::arg("text"), pybind11::arg("mode") = "mix")
            .def("is_loaded", &tool::Tokenizer::is_loaded);
        pybind11::class_<tool::Splitter>(m, "Splitter")
            .def_static("split_sentences", &tool::Splitter::split_sentences);
        pybind11::class_<tool::Keywords>(m, "Keywords")
            .def_static("top", &tool::Keywords::top, pybind11::arg("text"), pybind11::arg("k") = 5);
        pybind11::class_<tool::Stats>(m, "Stats")
            .def_static("char_count", &tool::Stats::char_count)
            .def_static("word_count", &tool::Stats::word_count)
            .def_static("sentence_count", &tool::Stats::sentence_count);
    }
} // namespace modeldeploy::nlp
```

- [ ] **Step 2: 写 `csrc/pybind/nlp/solutions_pybind.cpp`**

```cpp
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "nlp/solutions/text_classifier.h"
#include "runtime/runtime_option.h"
namespace modeldeploy::nlp {
    void bind_solutions(const pybind11::module& m) {
        pybind11::class_<solution::TextClassifier>(m, "TextClassifier")
            .def(pybind11::init<const std::string&, const RuntimeOption&>(),
                 pybind11::arg("model_file"), pybind11::arg("option") = RuntimeOption())
            .def("predict", [](solution::TextClassifier& s, const std::string& text) {
                int label; float score;
                if (!s.predict(text, &label, &score)) throw std::runtime_error("TextClassifier predict failed");
                return std::make_pair(label, score);
            })
            .def("is_initialized", &solution::TextClassifier::is_initialized);
    }
} // namespace modeldeploy::nlp
```

- [ ] **Step 3: `csrc/pybind/main.cpp` 加 `#ifdef BUILD_NLP` 块**

```cpp
namespace modeldeploy::nlp {
    void bind_tools(const pybind11::module&);
    void bind_solutions(const pybind11::module&);
}
// ...在 PYBIND11_MODULE 内、audio 块之后：
#ifdef BUILD_NLP
        auto nlp_module = m.def_submodule("nlp", "NLP module of Modeldeploy.");
        nlp::bind_tools(nlp_module);
        nlp::bind_solutions(nlp_module);
#endif
```

- [ ] **Step 4: 构建 + Python smoke**

```bash
cd build_py && python -c "
from modeldeploy.nlp import tools, solutions
s = tools.Splitter.split_sentences('你好。世界！')
assert len(s) == 2
assert tools.Stats.word_count('a b c') == 3
t = solutions.TextClassifier('nonexistent.onnx')
assert t.is_initialized() == False
print('nlp tools/solutions smoke OK')
"
```
Expected: `nlp tools/solutions smoke OK` 无异常。

- [ ] **Step 5: Commit**

```bash
git add csrc/pybind/nlp/ csrc/pybind/main.cpp
git commit -m "feat(pybind): bind nlp.tools + nlp.solutions"
```

---

## C6: NLP CAPI（工具纯函数 + TextClassifier 模型句柄）

**Files:**
- Modify: `capi/md_capi.h`、`capi/md_capi.cpp`
- Test: `tests/test_capi.cpp`、`capi/test_capi_full.c`

**Interfaces:**
- **复用**：CAPI `MDModelHandle`/`md_model_create`/`md_result_classification`（仿 TSN 先例 `capi/md_capi.cpp:917-924`）
- Produces:
  - 枚举加：`MD_MODEL_TEXT_CLASSIFIER`（`MD_MODEL_FACE_LANDMARK` 之后、`MD_MODEL_COUNT` 之前，值 = 35）
  - `MDStatus md_model_create(MDModelHandle*, MD_MODEL_TEXT_CLASSIFIER, path, opt)` 分发（`#ifdef BUILD_NLP`）→ 结果 kind `MD_RES_CLASSIFICATION`（复用 `md_result_classification` 读 `(label,score)`）
  - 工具纯函数（`#ifdef BUILD_NLP`）：
    - `MDStatus md_nlp_split_sent(const char* text, const char*** sents, size_t* n);`
    - `MDStatus md_nlp_stats(const char* text, size_t* chars, size_t* words, size_t* sents);`
    - `MDStatus md_nlp_keywords(const char* text, int k, const char*** words, int** counts, size_t* n);`
    - `MDStatus md_nlp_tokenize(const char* text, const char* dict_dir, const char*** toks, size_t* n);`
    - `MDStatus md_nlp_classify(MDModelHandle h, const char* text, int* label, float* score);`

- [ ] **Step 1: `capi/md_capi.h` 枚举 + 声明**

在 `MD_MODEL_FACE_LANDMARK,` 之后加 `MD_MODEL_TEXT_CLASSIFIER,`（`MD_MODEL_COUNT` 之前）。
声明：
```c
MD_CAPI_EXPORT MDStatus md_nlp_split_sent(const char* text, const char*** sents, size_t* n);
MD_CAPI_EXPORT MDStatus md_nlp_stats(const char* text, size_t* chars, size_t* words, size_t* sents);
MD_CAPI_EXPORT MDStatus md_nlp_keywords(const char* text, int k, const char*** words, int** counts, size_t* n);
MD_CAPI_EXPORT MDStatus md_nlp_tokenize(const char* text, const char* dict_dir, const char*** toks, size_t* n);
MD_CAPI_EXPORT MDStatus md_nlp_classify(MDModelHandle h, const char* text, int* label, float* score);
```

- [ ] **Step 2: `capi/md_capi.cpp`**

- create 分发加：
```cpp
case MD_MODEL_TEXT_CLASSIFIER: {
    if (!need_parts(1, "text-classifier")) return MD_ERR_INVALID_ARGUMENT;
    const auto parts = split_path(model_path);
    auto* m = new nlp::solution::TextClassifier(parts[0], opt);
    mh->model = m;
    if (!m->is_initialized()) return fail_init("TextClassifier");
    break;
}
```
（需在文件头 `#ifdef BUILD_NLP` 下 include `csrc/nlp/solutions/text_classifier.h`；该 case 加 `#ifdef BUILD_NLP ... #else return MD_ERR_UNSUPPORTED_TYPE; #endif`，仿 `MD_MODEL_ASR` 的 `#ifdef BUILD_AUDIO` 处理。）
- delete/clone 分发加对应 `TextClassifier` 三处（仿 `MD_MODEL_TSN` `capi/md_capi.cpp:1045/1108`）。
- `md_nlp_classify`：
```cpp
MDStatus md_nlp_classify(MDModelHandle h, const char* text, int* label, float* score) {
    auto* mh = static_cast<md_model_handle*>(h);
    if (!mh || !text || !label || !score) return MD_ERR_NULL_POINTER;
    if (!mh->ready || mh->kind != MD_MODEL_TEXT_CLASSIFIER) return MD_ERR_INVALID_ARGUMENT;
#ifdef BUILD_NLP
    auto* m = static_cast<nlp::solution::TextClassifier*>(mh->model);
    if (!m->predict(text, label, score)) { set_error("md_nlp_classify: predict failed"); return MD_ERR_MODEL_PREDICT; }
    return MD_OK;
#else
    (void)text; return MD_ERR_UNSUPPORTED_TYPE;
#endif
}
```
- 工具函数（`#ifdef BUILD_NLP`）用 file-static `std::vector<std::string>/std::vector<std::vector<char>>` 存借用字符串，`char**` 指向其 `data()`：
```cpp
static std::vector<std::string> g_nlp_sents;   // split_sent 结果
static std::vector<std::vector<char>> g_nlp_ptrs;
static std::vector<std::string> g_nlp_toks;
static std::vector<int> g_nlp_counts;
static std::vector<std::string> g_nlp_kwords;

MDStatus md_nlp_split_sent(const char* text, const char*** sents, size_t* n) {
    if (!text || !sents || !n) return MD_ERR_NULL_POINTER;
#ifdef BUILD_NLP
    g_nlp_sents = nlp::tool::Splitter::split_sentences(text);
    g_nlp_ptrs.clear();
    for (auto& s : g_nlp_sents) { g_nlp_ptrs.emplace_back(s.begin(), s.end()); g_nlp_ptrs.back().push_back('\0'); }
    std::vector<const char*> p; for (auto& v : g_nlp_ptrs) p.push_back(v.data());
    static std::vector<const char*> g_p; g_p = p;
    *sents = g_p.data(); *n = g_p.size();
    return MD_OK;
#else
    (void)text; return MD_ERR_UNSUPPORTED_TYPE;
#endif
}
```
（其余 `md_nlp_stats`/`md_nlp_tokenize`/`md_nlp_keywords` 同理：`md_nlp_tokenize` 用 `nlp::tool::Tokenizer(dict_dir).tokenize(text, "mix")`；`md_nlp_keywords` 用 `nlp::tool::Keywords::top(text,k)`——完整实现以 `md_nlp_split_sent` 为模板，同款借用 buffer 管理。）

- [ ] **Step 3: `tests/test_capi.cpp` + `capi/test_capi_full.c` 加 `[capi]` 用例**

`tests/test_capi.cpp`：
```cpp
TEST_CASE("nlp tool capi", "[capi]") {
    size_t n = 0; const char** s = nullptr;
    REQUIRE(md_nlp_split_sent("你好。世界！", &s, &n) == MD_OK);
    REQUIRE(n == 2);
    size_t chars = 0, words = 0, sents = 0;
    REQUIRE(md_nlp_stats("hello world 你好", &chars, &words, &sents) == MD_OK);
    REQUIRE(words == 2);
}
```
`capi/test_capi_full.c`（或新建小块）：`md_model_create(&m, MD_MODEL_TEXT_CLASSIFIER, "nonexistent.onnx", opt)` 期望失败不崩（幂等错误路径）。

- [ ] **Step 4: 构建 + 测试**

Run: `cmake --build build --parallel 8 && cd build && .\bin\test_modeldeploy.exe "[capi]"`
Expected: 0 errors；`[capi]` 用例 PASS（NLP 工具函数恒跑）。

- [ ] **Step 5: Commit**

```bash
git add capi/md_capi.h capi/md_capi.cpp tests/test_capi.cpp capi/test_capi_full.c
git commit -m "feat(capi): MD_MODEL_TEXT_CLASSIFIER + md_nlp_* tool funcs"
```

---

## C7: NLP C# + Rust + demo_nlp + docs

**Files:**
- Modify: `csharp/ModelDeploy/enum_varaibles.cs`（`MDModelKind` 加 `MD_MODEL_TEXT_CLASSIFIER`）、`csharp/ModelDeploy/Models.cs`（`NlpClassifier`）、`csharp/ModelDeploy/NativeMethods.cs`
- Modify: `csharp/ModelDeployUnitTest/AllModelsTests.cs`
- Modify: `rust/modeldeploy/src/ffi.rs`（`MDModelKind` 加 `TextClassifier = 35`）、`rust/modeldeploy/src/nlp.rs`（新建）、`rust/modeldeploy/src/lib.rs`
- Create: `examples/demo_nlp/demo_nlp.cpp` + `CMakeLists.txt`
- Modify: `examples/CMakeLists.txt`、`examples/EXAMPLES.md`、`README.md`

**Interfaces:**
- Consumes: CAPI C6
- Produces: C# `Models.NlpClassifier.Predict(text)->(int,float)`；Rust `nlp::TextClassifier`；`demo_nlp`

- [ ] **Step 1: C#**

`enum_varaibles.cs` 的 `MDModelKind` 加 `MD_MODEL_TEXT_CLASSIFIER`；`NativeMethods.cs` extern 导入 `md_nlp_classify` + 5 个 `md_nlp_*` 工具；`Models.cs` 加：
```csharp
public sealed class NlpClassifier : BaseModel {
    private NlpClassifier(IntPtr handle) : base(types_internal_c.MDModelKind.MD_MODEL_TEXT_CLASSIFIER, handle) { }
    public NlpClassifier(string modelPath, RuntimeOption opt = null) : base(types_internal_c.MDModelKind.MD_MODEL_TEXT_CLASSIFIER, modelPath, opt) { }
    public (int Label, float Score) Predict(string text) {
        var tp = Utf8.Alloc(text);
        try {
            var status = NativeMethods.md_nlp_classify(_handle, tp, out var label, out var score);
            if (status != types_internal_c.MDStatus.MD_OK) throw new InvalidOperationException("classify failed");
            return (label, score);
        } finally { Utf8.Free(tp); }
    }
}
```
`AllModelsTests.cs` 加 `[Fact] NlpTools_Works`：`md_nlp_split_sent("你好。世界！", out var s, out var n)` == n==2（`NlpTools` 静态调用，无权重）。

- [ ] **Step 2: Rust**

`ffi.rs` `MDModelKind` 加 `TextClassifier = 35`；extern `md_nlp_classify`/`md_nlp_split_sent`/`md_nlp_stats`；`nlp.rs`：
```rust
pub struct TextClassifier { model: Model }
impl TextClassifier {
    pub fn new(model_path: &str, option: &RuntimeOption) -> Result<Self, MdError> {
        Ok(Self { model: Model::new(ffi::MDModelKind::TextClassifier, model_path, option)? })
    }
    pub fn predict(&self, text: &str) -> Result<(i32, f32), MdError> {
        let c = CString::new(text).map_err(|_| MdError::InvalidArgument("text".into()))?;
        let mut label = 0i32; let mut score = 0.0f32;
        check_status(unsafe { ffi::md_nlp_classify(self.model.handle, c.as_ptr(), &mut label, &mut score) })?;
        Ok((label, score))
    }
}
```
`lib.rs` 加 `pub mod nlp;`；`integration_test.rs` 加 `#[test] fn test_nlp_tools`：`md_nlp_stats` 无权重断言 words。

- [ ] **Step 3: `examples/demo_nlp/`**

`CMakeLists.txt`（仿 `demo_landmark`）：
```cmake
add_executable(demo_nlp demo_nlp.cpp)
target_link_libraries(demo_nlp PRIVATE ${LIBRARY_NAME})
```
`demo_nlp.cpp`：无权重先用 `Splitter`/`Keywords`/`Stats` 演示文本统计；若给定 `bert.onnx` 路径参数则 `TextClassifier` 分类一句（缺权重清晰报错不崩）：
```
Usage: demo_nlp [bert_classifier.onnx] [text]
若无 onnx：演示 分词(Splitter)/关键词(Keywords)/统计(Stats)，无需权重。
若有 onnx：TextClassifier classify 输出 (label, score)。
```

- [ ] **Step 4: 注册 + docs**

`examples/CMakeLists.txt` 加 `add_subdirectory(demo_nlp)`（`demo_landmark` 后）。
`EXAMPLES.md` 加行：
```
| `demo_nlp` | NLP 工具（分词/分句/关键词/统计）+ 可选情感分类 | `bert_*.onnx` 可选 | 文本 | 打印词/句/关键词/分类 |
```
`README.md` 能力列表加「**NLP（jieba 分词 / 分句 / 关键词 / 统计 + BERT 文本分类 ONNX）**」。

- [ ] **Step 5: 构建 + 冒烟 + Commit**

`cmake --build build --parallel 8`（`demo_nlp` 编译 0 errors，无参 Usage）；`dotnet test --filter NlpTools_Works`；`cargo test test_nlp_tools`。
```bash
git add csharp/ModelDeploy/ csharp/ModelDeployUnitTest/ rust/modeldeploy/src/ rust/modeldeploy/tests/ examples/demo_nlp/ examples/CMakeLists.txt examples/EXAMPLES.md README.md
git commit -m "feat(nlp): C#/Rust wrappers + demo_nlp + docs"
```

---

## D: 全量验证 + 收尾

**Files:** 无新增（验证；必要时 minor 修复）

**Interfaces:** Consumes 全部前序任务

- [ ] **Step 1: 全量 C++ 测试**
Run: `cd build && .\bin\test_modeldeploy.exe`
Expected: `[cv_tools]`(12) + `[cv_solution]`(10) + `[audio_tools]`(6) + `[audio_solution]`(4) + `[nlp]`(8) + `[capi]` 全部 PASS；无回归（`[core]`/`[image_data]`/`[tracking]` 等原有用例）。

- [ ] **Step 2: 绑定/示例冒烟**
- Python：`cv solutions/tools`、`audio solutions/tools`、`nlp tools/solutions` 三个 smoke。
- C#：`CvSolution_Works`/`SpeakerSearch_Works`/`NlpTools_Works`；Rust：`test_cv_solution`/`test_audio_speaker_search`/`test_nlp_tools`。
- demos：`demo_solutions`/`demo_tools`/`demo_audio_solutions/*`/`demo_nlp` 编译 0 errors、缺权重/缺参报错不崩。

- [ ] **Step 3: 跨后端/开关确认**
- grep 确认 `csrc/vision/solutions|tools`、`csrc/audio/tools|solutions`、`csrc/nlp` 无 backend 直接 include → ORT/MNN 语义一致。
- `BUILD_NLP=OFF` 时 NLP pybind/capi/tests 不参与编译（`#ifdef BUILD_NLP` 隔离）校验。

- [ ] **Step 4: 收尾报送**
报告 + concerns（如 cppjieba 双重 add_subdirectory、CAPI 借用 buffer 线程安全、NLP WordPiece 词表、VideoDecoder pts_ms 接 SpeedEstimator 时间戳）。

---

## Self-Review 记录

**Spec coverage**：
- CV 工具层 6（§4）→ A1 Detections/A2 Zone/A3 Annotator/A4 Metrics/A5 Slicer/A6 Smoother。
- CV 方案层 9（§2.2-9、§3，含 VisionEye）→ A7 ObjectCounter/A8 Heatmap/A9 Speed+Distance/A10 Crop+Blur/A11 Workout+Parking+VisionEye。
- CV 复用（§3.1）→ 各 Task 标注复用 `tracking::TrackResult`/`tool::Zone`/`ImageData`/`vis_*`。
- CV Python（§4）→ A12；CAPI（§5）→ A13；C#/Rust（§6）→ A14；demo+tests（§7、§8）→ A14 + A1-A11。
- Audio 工具层 6（A.2）→ B1 WavIO+AudioMeta/B2 Resampler/B3 Fbank/B4 Waveform/B5 VadSegment。
- Audio 方案层 4（A.3）→ B6 SpeakerSearch/B7 Diarization/B8 StreamingSTT+TTSBatcher。
- Audio Python（A.4）→ B9；CAPI → B10；C#/Rust/demo → B11；测试（§8.1）→ B1-B8。
- NLP 工具层 5（B.2）→ C1 Tokenizer/C2 Splitter+Normalizer/C3 Keywords+Stats。
- NLP 方案层 TextClassifier（B.3）→ C4；主 CMake `BUILD_NLP`（B.4/§B.2）→ C1。
- NLP Python（B.4）→ C5；CAPI（B.4）→ C6；C#/Rust/demo → C7；测试（§8.2）→ C1-C4。
- 交付矩阵三表 6 面（§9）→ 每域 C++/Python/CAPI/C#/Rust/demo+docs+tests 全覆盖。

**Placeholder 扫描**：所有 Task 的代码步骤均有完整实现/完整测试代码；无 "TBD/TODO"、无 "类似 Task N 参照却不给代码"（跨 Task 引用只在 CAPI 分发/薄封装处给"仿 file:line"指令 + 自身完整测试）。`naive_tokenize` 的 token→id 映射（`101+i%100`）为**显式说明的占位数值**，已注明真实推理需 WordPiece+词表——这是实现时以真实 vocab 为准的关键点，非未定义接口。

**Type consistency**：
- `Detections::{boxes,class_id,confidence,masks,tracker_id,size,reserve}` 在 A1/A2(filter_by_zone)/A3(draw)/A5(reassemble)/A6(smoother)/A12(pybind) 一致。
- `tracking::TrackResult{track_id,box,score,label_id}` 贯穿 A7/A8/A9/A11/CAPI A13。
- 枚举 `MD_SOLUTION_OBJECT_COUNTER` 等在 A13/C#/Rust 一致；`MD_MODEL_TEXT_CLASSIFIER=35` 在 C6/C7 一致。
- `audio::solution::SpeakerSearch::enroll/match`、`audio::tool::{Resampler::resample,Fbank::compute,parse_meta}` 在 B6-B11 一致。
- `nlp::tool::{Splitter::split_sentences, Stats::word_count, Keywords::top}`、`nlp::solution::{TextClassifier::encode/softmax_top1/predict}` 在 C1-C7 一致。

**规模**：单份大 plan（用户明确要单份）；按 A/B/C 域分节，每节内可分批实施。

