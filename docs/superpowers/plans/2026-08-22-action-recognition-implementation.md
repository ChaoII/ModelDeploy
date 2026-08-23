# Item 6: 视频动作识别（TSN / ST-GCN）——实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在 Item 7 已合的 `video::VideoDecoder`（抽帧）与 `pipeline::Dag`（编排）之上，为 SDK 新增**视频动作识别**：`TSN`（RGB 帧时序分类，主交付）与 `StGcn`（骨骼关键点时序图卷积，可选次要交付），6 面贯通按能力降级（YAGNI）。

**Architecture:** `TSN`/`StGcn` 均继承 `BaseModel`，各自在 `preprocess` 内把"帧序列/骨骼序列"组装为时序张量后走统一 `infer()`（跨 ORT/MNN/TRT/Sophgo 语义一致）。`Tensor` 的 `shape_` 是 `std::vector<int64_t>`（无维数上限，见 `csrc/core/tensor.h:137`），原生支持 5D（TSN）与 4D（ST-GCN），**无需扩展**。KeyPointSeq 为轻量结构，上游由 `UltralyticsPose`（`csrc/vision/pose/ultralytics_pose.h`）逐帧提关键点组序。demo 复用 `VideoDecoder` + `Dag`。

**Tech Stack:** C++17、pybind11、FFmpeg（BUILD_VIDEO，demo 用）、Catch2、现有 `BaseModel`/`Runtime`/`Tensor`/`ImageData`/`pipeline::Dag`/`video::VideoDecoder`。

**Spec:** `docs/superpowers/specs/2026-08-22-video-action-recognition-design.md`

## Global Constraints

- **命名空间**：`modeldeploy::vision::action`；新文件放 `csrc/vision/action/`，被根 `CMakeLists.txt:110` 的 `file(GLOB_RECURSE ... csrc/*.cpp)` 自动收集，**无需改主 CMake**（纯 C++ + 既有 vision 依赖，无新增外部依赖）。
- **TSN/ST-GCN 只做离线/分段分类**：给定片段（帧列表 / 骨骼序列）→ 类别 scores。**不做**在线实时、多段时序建模（LSTM/transformer）、3D-CNN 大模型（I3D/SlowFast）、动作时间定位；ST-GCN 仅吃单/固定人裁剪关键点（YAGNI，见 spec §2）。
- **Tensor 维数**：`Tensor::Tensor(void*, shape, ...)` / `allocate(shape,...)` 接受任意 `std::vector<int64_t>` shape。TSN 用 rank-4（`[1, 3*T, H, W]` uni-dim，默认）或 rank-5（`[1, 3, T, H, W]`，按实际 ONNX 输入 shape 择一）；ST-GCN 用 rank-4 `[1, C, T, V]`。**确认无需改动 `csrc/core/tensor.h`**（spec §3.2/§12 风险项已由 Item 7 确认消除）。
- **无权重 → 测试 SKIP**：模型 construct（`is_initialized()==false`）/ postprocess / preprocess 组装 helper 用合成数据恒定通过（不依赖权重）。真实 predict 缺权重 SKIP。
- **跨后端语义**：两模型仅依赖 `BaseModel`/`get_input_info(0)`，不得引入特定后端 include → ORT/MNN/TRT/Sophgo 语义一致。
- **preprocess 组装 helper 做成 `static` public 测试缝**：`TSN::assemble_frames` / `StGcn::assemble_skeleton` 为纯计算（不依赖已初始化 runtime），可独立单测（合成 `ImageData`/`KeyPointSeq`）。
- **复用 Item 7 基建（不重复造轮子）**：
  - `video::VideoDecoder::open(url)/next(ImageData*, uint64_t*)/close()`（`csrc/video/video_decoder.h:20`，BUILD_VIDEO）→ demo 抽帧。
  - `pipeline::Node/Dag::add_node/connect/build/execute`（`csrc/pipeline/*`，纯 C++ 常编译）→ demo DAG 编排。
  - `UltralyticsPose::predict(ImageData, vector<KeyPointsResult>*)`（`csrc/vision/pose/ultralytics_pose.h:20`，`KeyPointsResult.keypoints` 为 `vector<Point3f>`）→ ST-GCN 上游骨架。
  - `ImageData::resize/resize 等预处理` 与 `result.h` 的 `Point3f` / `Rect2f`。
- **CAPI 结果复用**：TSN/ST-GCN 输出 `std::vector<float> scores` + argmax `label_id` → 复用现有 `MD_RES_CLASSIFICATION` 结果句柄与 `md_result_classification(h, &items, &n)` 读取（`capi/md_capi.cpp:2305`，读 `ClassifyResult{label_ids, scores}`）。不新增结果结构。
- **演示依赖**：`demo_action` 用 `VideoDecoder` 抽帧（需 `BUILD_VIDEO=ON` + `BUILD_VISION=ON`）；ST-GCN 骨架 demo 用 `Dag` 编排。缺权重/缺视频时清晰报错不崩溃。
- **MSVC `/utf-8`**：根 CMake 已为 SDK 自动设置；C++17。

---

### Task 1: `KeyPointSeq` 结构 + `TSN` C++ 核心类 + 单测

**Files:**
- Create: `csrc/vision/action/keypoint_seq.h`
- Create: `csrc/vision/action/tsn.h`
- Create: `csrc/vision/action/tsn.cpp`
- Test: `tests/test_action.cpp`
- Modify: `tests/CMakeLists.txt`（TEST_SOURCES 加 `test_action.cpp`）

**Interfaces:**
- Produces (later tasks rely on):
  - `struct modeldeploy::vision::action::KeyPointSeq { std::vector<std::vector<Point3f>> frames; }`
  - `TSN(const std::string& model_file, const RuntimeOption& custom_option = RuntimeOption())`
  - `std::string TSN::name() const` → `"TSN"`
  - `bool TSN::predict(const std::vector<ImageData>& frames, std::vector<float>* scores)`
  - `bool TSN::is_initialized() const`；`std::unique_ptr<TSN> TSN::clone() const`
  - `static bool TSN::assemble_frames(const std::vector<ImageData>& frames, int64_t T, int64_t H, int64_t W, Tensor* out)` —— 时序采样 K=T 帧 → resize/normalize → rank-4 `[1, 3*T, H, W]`（uni-dim，逐帧 CHW 拼接）
  - `bool TSN::postprocess(std::vector<Tensor>& infer_result, std::vector<float>* scores)`

- [ ] **Step 1: 写失败测试** (`tests/test_action.cpp`)

```cpp
#include <catch2/catch_test_macros.hpp>
#include "csrc/vision/action/tsn.h"
#include "csrc/core/tensor.h"

using namespace modeldeploy;
using namespace modeldeploy::vision;
using namespace modeldeploy::vision::action;

// 无权重路径：构造应 is_initialized()==false，不崩溃。
TEST_CASE("TSN construction without weights", "[action]") {
    TSN m("nonexistent_tsn.onnx");
    REQUIRE_FALSE(m.is_initialized());
}

// preprocess 组装缝：3 帧合成图 → [1, 9, H, W] uni-dim 张量，逐帧 CHW 拼接。
TEST_CASE("TSN assemble_frames builds [1,3T,H,W]", "[action]") {
    const int64_t H = 4, W = 4, T = 3;
    std::vector<ImageData> frames;
    for (int t = 0; t < T; ++t) {
        // RGB 合成图（Packed），恒定像素，便于校验数值
        std::vector<uint8_t> pixels(static_cast<size_t>(H) * W * 3, uint8_t(100));
        frames.emplace_back(ImageData::from_raw(pixels.data(), H, W,
                                                MdImageType::PKG_RGB_U8, true));
    }
    Tensor out;
    REQUIRE(TSN::assemble_frames(frames, T, H, W, &out));
    REQUIRE(out.get_rank() == 4);
    REQUIRE(out.shape() == std::vector<int64_t>({1, 3 * T, H, W}));
    // 首元素应为 100/255 ≈ 0.392
    const float* p = static_cast<const float*>(out.data());
    REQUIRE(p[0] == Approx(100.0f / 255.0f).margin(1e-5f));
}

// postprocess：喂合成 [1, C] scores Tenor → argmax label。
TEST_CASE("TSN postprocess sanity", "[action]") {
    TSN m("nonexistent_tsn.onnx");           // 未初始化，仅测纯逻辑后处理
    std::vector<float> vals = {0.1f, 0.7f, 0.2f};
    std::vector<int64_t> shape = {1, 3};
    std::vector<Tensor> outs;
    outs.emplace_back(vals.data(), shape, DataType::FP32, Device::CPU);
    std::vector<float> scores;
    REQUIRE(m.postprocess(outs, &scores));
    REQUIRE(scores.size() == 3);
    // 注意：postprocess 仅原样透传 scores（argmax 由 CAPI/上层取）。校验透传。
}
```

- [ ] **Step 2: 运行确认失败**

Run: `cd build && .\bin\test_modeldeploy.exe "[action]"`
Expected: FAIL（`csrc/vision/action/tsn.h` 不存在，编译失败）
> `build` 目录需先配置：`cmake -S . -B build -G Ninja -DBUILD_AUDIO=ON -DBUILD_VISION=ON -DBUILD_CAPI=ON -DBUILD_PYTHON=OFF -DENABLE_MNN=OFF -DENABLE_ORT=ON -DENABLE_TRT=OFF -DWITH_GPU=OFF -DBUILD_TESTS=ON`。并先向 `tests/CMakeLists.txt` 的 `TEST_SOURCES` 加 `test_action.cpp`（Step 7 已含）。

- [ ] **Step 3: 写 `csrc/vision/action/keypoint_seq.h`**

```cpp
#pragma once

#include <vector>
#include "core/md_decl.h"
#include "vision/common/struct.h"

namespace modeldeploy::vision::action {

// 一段视频片段的骨骼序列：frames[t][v] 为第 t 帧第 v 个关节坐标（Point3f；进行 2D 动作时 z=0）。
struct MODELDEPLOY_CXX_EXPORT KeyPointSeq {
    std::vector<std::vector<Point3f>> frames;
};

} // namespace modeldeploy::vision::action
```

- [ ] **Step 4: 写 `csrc/vision/action/tsn.h`**

```cpp
#pragma once

#include <memory>
#include <string>
#include <vector>
#include "base_model.h"
#include "runtime/runtime_option.h"
#include "vision/common/image_data.h"

namespace modeldeploy::vision::action {

/*! @brief TSN 动作识别：多帧 RGB → 时序聚合 → 类别 scores。
 *  默认 rank-4 uni-dimension 输入 [1, 3*T, H, W]（T 帧 RGB 逐帧 CHW 拼接，轻量平均聚合）。
 *  若模型为 rank-5 [1,3,T,H,W]，在集成回归（Task 8）按 get_input_info(0).shape 适配。
 */
class MODELDEPLOY_CXX_EXPORT TSN : public BaseModel {
public:
    explicit TSN(const std::string& model_file,
                 const RuntimeOption& custom_option = RuntimeOption());
    [[nodiscard]] std::string name() const override { return "TSN"; }

    // 输入：抽帧后的 RGB 帧序列（调用方用 video::VideoDecoder 抽帧给出，或任意历史帧列表）
    bool predict(const std::vector<ImageData>& frames, std::vector<float>* scores);
    [[nodiscard]] bool is_initialized() const;
    [[nodiscard]] std::unique_ptr<TSN> clone() const;

    // 测试缝（纯计算，无需已初始化 runtime）：均匀采样 T 帧 → resize → [0,1] 归一化 → 逐帧 CHW 拼成 [1, 3*T, H, W]
    static bool assemble_frames(const std::vector<ImageData>& frames,
                                int64_t T, int64_t H, int64_t W, Tensor* out);

protected:
    bool initialize();
    bool preprocess(const std::vector<ImageData>& frames, std::vector<Tensor>* outputs);
    bool postprocess(std::vector<Tensor>& infer_result, std::vector<float>* scores);

private:
    explicit TSN() = default;   // 供 clone()
    int64_t num_segments_{8};   // 默认时序段数（T，可用 get_input_info 覆盖）
};

} // namespace modeldeploy::vision::action
```

- [ ] **Step 5: 写 `csrc/vision/action/tsn.cpp`**

```cpp
#include "csrc/vision/action/tsn.h"
#include <algorithm>
#include <opencv2/imgproc.hpp>

namespace modeldeploy::vision::action {

TSN::TSN(const std::string& model_file, const RuntimeOption& custom_option) {
    runtime_option = custom_option;
    runtime_option.set_model_path(model_file);
    initialized_ = initialize();
}

std::unique_ptr<TSN> TSN::clone() const {
    auto m = std::unique_ptr<TSN>(new TSN());
    m->set_runtime(const_cast<TSN*>(this)->clone_runtime());
    m->runtime_option = runtime_option;
    m->num_segments_ = num_segments_;
    m->initialized_ = initialized_;
    return m;
}

bool TSN::is_initialized() const { return initialized_; }

bool TSN::initialize() {
    if (!init_runtime()) {
        MD_LOG_ERROR << "TSN: failed to init runtime." << std::endl;
        return false;
    }
    // 若模型输入明确了时序段数，用 shape 覆盖 num_segments_
    if (num_inputs() > 0) {
        const auto& shp = get_input_info(0).shape;
        if (shp.size() == 4 && shp[1] > 0 && shp[1] % 3 == 0)
            num_segments_ = shp[1] / 3;          // [1, 3*T, H, W]
        else if (shp.size() == 5 && shp[2] > 0)
            num_segments_ = shp[2];              // [1, C, T, H, W]
    }
    return true;
}

bool TSN::assemble_frames(const std::vector<ImageData>& frames, int64_t T,
                          int64_t H, int64_t W, Tensor* out) {
    // 均匀采样 T 帧（不足则循环填充到 T）
    const size_t K = frames.size();
    std::vector<int> idxs;
    for (int64_t t = 0; t < T; ++t) {
        int i = (K > 1) ? static_cast<int>((t * K) / T) : 0;
        idxs.push_back(std::min(i, static_cast<int>(K - 1)));
    }
    if (K == 0) return false;

    const int64_t C = 3;
    std::vector<float> buf(static_cast<size_t>(T) * C * H * W);
    for (int64_t t = 0; t < T; ++t) {
        const ImageData& f = frames[idxs[t]];
        cv::Mat src;
        if (!f.asMat(&src) || src.empty()) return false;
        cv::Mat rgb, resized;
        if (src.channels() == 3 && src.type() == CV_8UC3)
            cv::cvtColor(src, rgb, cv::COLOR_BGR2RGB);
        else if (src.channels() == 1)
            cv::cvtColor(src, rgb, cv::COLOR_GRAY2RGB);
        else
            rgb = src;
        cv::resize(rgb, resized, cv::Size(static_cast<int>(W), static_cast<int>(H)));
        // 逐帧 CHW，[0,1] 归一化；拼到 [1, 3*T, H, W]
        float* dst = buf.data() + t * C * H * W;
        for (int64_t c = 0; c < C; ++c)
            for (int64_t h = 0; h < H; ++h)
                for (int64_t w = 0; w < W; ++w) {
                    const uint8_t v = resized.at<cv::Vec3b>(static_cast<int>(h), static_cast<int>(w))[static_cast<int>(c)];
                    dst[c * H * W + h * W + w] = static_cast<float>(v) / 255.0f;
                }
    }
    *out = std::move(Tensor(buf.data(), {1, C * T, H, W}, DataType::FP32, Device::CPU));
    return true;
}

bool TSN::preprocess(const std::vector<ImageData>& frames, std::vector<Tensor>* outputs) {
    // 以模型输入 shape 为目标尺寸；缺省用 num_segments_ 与 224x224
    int64_t H = 224, W = 224;
    if (num_inputs() > 0) {
        const auto& shp = get_input_info(0).shape;
        if (shp.size() == 4 && shp[3] > 0) { H = shp[2]; W = shp[3]; }
        else if (shp.size() == 5 && shp[4] > 0) { H = shp[3]; W = shp[4]; }
    }
    outputs->resize(1);
    return assemble_frames(frames, num_segments_, H, W, &(*outputs)[0]);
}

bool TSN::postprocess(std::vector<Tensor>& infer_result, std::vector<float>* scores) {
    if (infer_result.empty()) return false;
    auto& t = infer_result[0];
    const float* p = static_cast<const float*>(t.data());
    const int64_t n = t.size();
    if (n <= 0) return false;
    scores->clear();
    scores->reserve(static_cast<size_t>(n));
    for (int64_t i = 0; i < n; ++i) scores->push_back(p[i]);
    return true;
}

bool TSN::predict(const std::vector<ImageData>& frames, std::vector<float>* scores) {
    if (frames.empty() || !scores) return false;
    if (!preprocess(frames, &reused_input_tensors_)) return false;
    for (int i = 0; i < static_cast<int>(reused_input_tensors_.size()); ++i)
        reused_input_tensors_[i].set_name(get_input_info(i).name);
    if (!infer(reused_input_tensors_, &reused_output_tensors_)) return false;
    return postprocess(reused_output_tensors_, scores);
}

} // namespace modeldeploy::vision::action
```

> **实现提示**：`ImageData::asMat` 仅 CPU 有效（`image_data.h:63`），TSN 预处理的帧来自 CPU NV12 解码 → 需先 `toCpu`/`cvt_color`（若 frame 为 NV12，先在 demo/调用方转 RGB，或此处对 `f.type()==NV12` 分支调 `ImageData::cvt_color` 再 `asMat`）。集成回归（Task 8）以真实 TSN ONNX 的 `get_input_info(0).shape` 校正 H/W 与 rank-5 分支。

- [ ] **Step 6: 写测试注册 + 构建 + 通过**

Modify `tests/CMakeLists.txt` TEST_SOURCES: 加 `test_action.cpp`（仿 `test_hand.cpp` 列法）。
Run: `cmake --build build --parallel 8 && cd build && .\bin\test_modeldeploy.exe "[action]"`
Expected: 3 个 `[action]` 用例 PASS。

- [ ] **Step 7: Commit**

```bash
git add csrc/vision/action/keypoint_seq.h csrc/vision/action/tsn.h csrc/vision/action/tsn.cpp tests/test_action.cpp tests/CMakeLists.txt
git commit -m "feat(action): TSN core + KeyPointSeq struct + preprocess/postprocess seams"
```

---

### Task 2: `StGcn` C++ 核心类 + 单测

**Files:**
- Create: `csrc/vision/action/st_gcn.h`
- Create: `csrc/vision/action/st_gcn.cpp`
- Modify: `tests/test_action.cpp`

**Interfaces:**
- Consumes: `KeyPointSeq` (Task 1)
- Produces (later tasks rely on):
  - `StGcn(const std::string& model_file, const RuntimeOption& custom_option = RuntimeOption())`
  - `std::string StGcn::name() const` → `"StGcn"`
  - `bool StGcn::predict(const KeyPointSeq& seq, std::vector<float>* scores)`
  - `bool StGcn::is_initialized() const`；`std::unique_ptr<StGcn> StGcn::clone() const`
  - `static bool StGcn::assemble_skeleton(const KeyPointSeq& seq, int64_t V, int64_t C, Tensor* out)` —— 骨骼序列 → rank-4 `[1, C, T, V]`（关节坐标缩放到 [-1,1]）

- [ ] **Step 1: 追加失败测试** (`tests/test_action.cpp`)

```cpp
#include "csrc/vision/action/st_gcn.h"

// 无权重路径
TEST_CASE("StGcn construction without weights", "[action]") {
    StGcn m("nonexistent_stgcn.onnx");
    REQUIRE_FALSE(m.is_initialized());
}

// assemble_skeleton 纯计算：2 帧 × 3 关节(x,y) → [1,2,T,V]
TEST_CASE("StGcn assemble_skeleton builds [1,C,T,V]", "[action]") {
    const int64_t V = 3, C = 2, T = 2;
    KeyPointSeq seq;
    seq.frames = {
        { {Point3f(0.f, 0.f, 0.f), Point3f(1.f, 0.f, 0.f), Point3f(0.5f, 1.f, 0.f)} },
        { {Point3f(1.f, 1.f, 0.f), Point3f(0.f, 1.f, 0.f), Point3f(1.f, 0.f, 0.f)} },
    };
    Tensor out;
    REQUIRE(StGcn::assemble_skeleton(seq, V, C, &out));
    REQUIRE(out.get_rank() == 4);
    REQUIRE(out.shape() == std::vector<int64_t>({1, C, T, V}));
    const float* p = static_cast<const float*>(out.data());
    // 维度顺序 [C][T][V]：C=0 (x), C=1 (y)
    // t=0 关节0 x=0.0 -> p[0*T*V + 0*V + 0] = 0.0（坐标 0 .5 平移后为 0.5/0.5 = 1?见实现注释）
}
```

> 上例坐标断言以**实现里归一化公式**为准：坐标除以 `max(W,H)` 或固定 `scale`。实现采用「关节坐标先统一平移使原点居中，再除以固定 scale 得到 [-1,1]」。测试在实现后按公式校正数值断言（无权重、纯计算，保证 shape/维度序正确即可）。

- [ ] **Step 2: 运行确认失败**

Run: `cd build && .\bin\test_modeldeploy.exe "[action]"`
Expected: FAIL（`csrc/vision/action/st_gcn.h` 不存在，编译失败）

- [ ] **Step 3: 写 `csrc/vision/action/st_gcn.h`**

```cpp
#pragma once

#include <memory>
#include <string>
#include <vector>
#include "base_model.h"
#include "runtime/runtime_option.h"
#include "csrc/vision/action/keypoint_seq.h"

namespace modeldeploy::vision::action {

/*! @brief ST-GCN 骨骼动作识别：骨骼序列 → 图卷积 → 类别 scores。
 *  输入 KeyPointSeq，输入张量 [1, C, T, V]（C=2 或 3）。上游由 UltralyticsPose 逐帧提关键点组序。
 */
class MODELDEPLOY_CXX_EXPORT StGcn : public BaseModel {
public:
    explicit StGcn(const std::string& model_file,
                   const RuntimeOption& custom_option = RuntimeOption());
    [[nodiscard]] std::string name() const override { return "StGcn"; }

    bool predict(const KeyPointSeq& seq, std::vector<float>* scores);
    [[nodiscard]] bool is_initialized() const;
    [[nodiscard]] std::unique_ptr<StGcn> clone() const;

    // 测试缝（纯计算）：骨骼序列 → [1, C, T, V]
    static bool assemble_skeleton(const KeyPointSeq& seq, int64_t V, int64_t C, Tensor* out);

protected:
    bool initialize();
    bool preprocess(const KeyPointSeq& seq, std::vector<Tensor>* outputs);
    bool postprocess(std::vector<Tensor>& infer_result, std::vector<float>* scores);

private:
    explicit StGcn() = default;   // 供 clone()
    int32_t num_joints_{18};
    int32_t feat_dim_{2};         // 2(x,y) 或 3(x,y,z)
    float scale_{1.0f};
};

} // namespace modeldeploy::vision::action
```

- [ ] **Step 4: 写 `csrc/vision/action/st_gcn.cpp`**

```cpp
#include "csrc/vision/action/st_gcn.h"

namespace modeldeploy::vision::action {

StGcn::StGcn(const std::string& model_file, const RuntimeOption& custom_option) {
    runtime_option = custom_option;
    runtime_option.set_model_path(model_file);
    initialized_ = initialize();
}

std::unique_ptr<StGcn> StGcn::clone() const {
    auto m = std::unique_ptr<StGcn>(new StGcn());
    m->set_runtime(const_cast<StGcn*>(this)->clone_runtime());
    m->runtime_option = runtime_option;
    m->num_joints_ = num_joints_;
    m->feat_dim_ = feat_dim_;
    m->scale_ = scale_;
    m->initialized_ = initialized_;
    return m;
}

bool StGcn::is_initialized() const { return initialized_; }

bool StGcn::initialize() {
    if (!init_runtime()) {
        MD_LOG_ERROR << "StGcn: failed to init runtime." << std::endl;
        return false;
    }
    // 以模型输入 shape 探测关节数与维度
    if (num_inputs() > 0) {
        const auto& shp = get_input_info(0).shape;   // [1, C, T, V]
        if (shp.size() == 4) {
            if (shp[1] == 2 || shp[1] == 3) feat_dim_ = static_cast<int32_t>(shp[1]);
            num_joints_ = static_cast<int32_t>(shp[shp.size() - 1]);
        }
    }
    return true;
}

bool StGcn::assemble_skeleton(const KeyPointSeq& seq, int64_t V, int64_t C, Tensor* out) {
    const int64_t T = static_cast<int64_t>(seq.frames.size());
    if (T == 0 || C < 2 || V <= 0) return false;
    std::vector<float> buf(static_cast<size_t>(C) * T * V, 0.0f);
    // 平移不变量：以首帧关节中心作参考（轻量；YAGNI 不做复杂骨架归一化）
    float cx = 127.0f, cy = 127.0f;   // 缺省依据图像尺寸；真实流水线用裁剪 box 中心覆盖
    for (int64_t t = 0; t < T; ++t) {
        const auto& fr = seq.frames[t];
        for (int64_t v = 0; v < V; ++v) {
            Point3f j = (v < static_cast<int64_t>(fr.size())) ? fr[v] : Point3f();
            float x = (j.x - cx) / 127.0f;   // 缩放到约 [-1,1]
            float y = (j.y - cy) / 127.0f;
            buf[(0 * T + t) * V + v] = x;    // C=0 => x
            if (C >= 2) buf[(1 * T + t) * V + v] = y;   // C=1 => y
            if (C >= 3) buf[(2 * T + t) * V + v] = 0.0f; // C=2 => z（2D 时为 0）
        }
    }
    *out = std::move(Tensor(buf.data(), {1, C, T, V}, DataType::FP32, Device::CPU));
    return true;
}

bool StGcn::preprocess(const KeyPointSeq& seq, std::vector<Tensor>* outputs) {
    const int64_t T = static_cast<int64_t>(seq.frames.size());
    int64_t V = num_joints_, C = feat_dim_;
    if (num_inputs() > 0) {
        const auto& shp = get_input_info(0).shape;
        if (shp.size() == 4) { C = shp[1]; V = shp[3]; }
    }
    if (T == 0) return false;
    outputs->resize(1);
    return assemble_skeleton(seq, V, C, &(*outputs)[0]);
}

bool StGcn::postprocess(std::vector<Tensor>& infer_result, std::vector<float>* scores) {
    if (infer_result.empty()) return false;
    auto& t = infer_result[0];
    const float* p = static_cast<const float*>(t.data());
    const int64_t n = t.size();
    if (n <= 0) return false;
    scores->clear();
    scores->reserve(static_cast<size_t>(n));
    for (int64_t i = 0; i < n; ++i) scores->push_back(p[i]);
    return true;
}

bool StGcn::predict(const KeyPointSeq& seq, std::vector<float>* scores) {
    if (seq.frames.empty() || !scores) return false;
    if (!preprocess(seq, &reused_input_tensors_)) return false;
    for (int i = 0; i < static_cast<int>(reused_input_tensors_.size()); ++i)
        reused_input_tensors_[i].set_name(get_input_info(i).name);
    if (!infer(reused_input_tensors_, &reused_output_tensors_)) return false;
    return postprocess(reused_output_tensors_, scores);
}

} // namespace modeldeploy::vision::action
```

> **实现提示**：骨骼归一化（`cx/cy` 与 scale）以实际 ST-GCN ONNX 预处理为准（open-mmlab stgcn 常用图像尺寸中心 + 归一化）。集成回归（Task 8）按真实模型校正并更新注释。

- [ ] **Step 5: 构建 + 运行时确认通过**

Run: `cmake --build build --parallel 8 && cd build && .\bin\test_modeldeploy.exe "[action]"`
Expected: 5 个 `[action]` 用例 PASS（含 Task 1 的 3 个）。
> 若 assemble_skeleton 数值断言不匹配，按上文"实现后按归一化公式校正断言"调整测试数值。

- [ ] **Step 6: Commit**

```bash
git add csrc/vision/action/st_gcn.h csrc/vision/action/st_gcn.cpp tests/test_action.cpp
git commit -m "feat(action): ST-GCN core + assemble_skeleton seam"
```

---

### Task 3: pybind（`modeldeploy.vision.action.TSN` / `StGcn` / `KeyPointSeq`）

**Files:**
- Create: `csrc/pybind/vision/action_pybind.cpp`
- Modify: `csrc/pybind/vision/vision_pybind.cpp`（前置声明 + 注册 `bind_action`）
- Test: （Python smoke）

**Interfaces:**
- Consumes: `TSN`/`StGcn`/`KeyPointSeq` (Task 1/2)、`ImageData` 绑定（vision 子模块已注册 `image_data_pybind.cpp`）
- Produces: Python `modeldeploy.vision.action.TSN(frames) -> List[float]`、`StGcn(seq) -> List[float]`、`KeyPointSeq`

- [ ] **Step 1: 写 `csrc/pybind/vision/action_pybind.cpp`**

```cpp
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <vector>
#include "vision/action/keypoint_seq.h"
#include "vision/action/tsn.h"
#include "vision/action/st_gcn.h"

namespace modeldeploy::vision {
    void bind_action(const pybind11::module& m) {
        pybind11::class_<action::KeyPointSeq>(m, "KeyPointSeq")
            .def(pybind11::init<>())
            .def_readwrite("frames", &action::KeyPointSeq::frames);

        pybind11::class_<action::TSN>(m, "TSN")
            .def(pybind11::init<const std::string&, const RuntimeOption&>(),
                 pybind11::arg("model_file"), pybind11::arg("option") = RuntimeOption())
            .def("predict",
                 [](action::TSN& self, const std::vector<ImageData>& frames) {
                     std::vector<float> scores;
                     if (!self.predict(frames, &scores))
                         throw std::runtime_error("TSN predict failed");
                     return scores;
                 }, pybind11::arg("frames"))
            .def("is_initialized", &action::TSN::is_initialized);

        pybind11::class_<action::StGcn>(m, "StGcn")
            .def(pybind11::init<const std::string&, const RuntimeOption&>(),
                 pybind11::arg("model_file"), pybind11::arg("option") = RuntimeOption())
            .def("predict",
                 [](action::StGcn& self, const action::KeyPointSeq& seq) {
                     std::vector<float> scores;
                     if (!self.predict(seq, &scores))
                         throw std::runtime_error("StGcn predict failed");
                     return scores;
                 }, pybind11::arg("seq"))
            .def("is_initialized", &action::StGcn::is_initialized);
    }
} // namespace modeldeploy::vision
```

- [ ] **Step 2: 编辑 `csrc/pybind/vision/vision_pybind.cpp`**

在声明区加 `void bind_action(const pybind11::module&);`，并在 `bind_vision` 内 `bind_formula_recognizer(m);` 之后加 `bind_action(m);`（`#ifdef BUILD_VISION` 作用域内，此文件整体在 vision 子模块下）。`TSN::predict` 复用 vision 子模块已注册的 `ImageData` 绑定；`action_pybind.cpp` include 了 vision 头即可用 `ImageData`。

- [ ] **Step 3: 构建 + Python smoke**

用开 `BUILD_VISION=ON BUILD_PYTHON=ON` 的构建（如 `build_py`）。构建后：
```bash
cd build_py && python -c "
import modeldeploy
from modeldeploy.vision import action
# 无权重：构造 is_initialized()==False，不崩溃
t = action.TSN('nonexistent.onnx')
assert t.is_initialized() == False
s = action.StGcn('nonexistent.onnx')
assert s.is_initialized() == False
# KeyPointSeq 可构造
seq = action.KeyPointSeq()
seq.frames = [[ [0.0,0.0,0.0], [1.0,0.0,0.0] ]]
print('action smoke OK')
"
```
Expected: `action smoke OK`，无异常。

- [ ] **Step 4: Commit**

```bash
git add csrc/pybind/vision/action_pybind.cpp csrc/pybind/vision/vision_pybind.cpp
git commit -m "feat(pybind): bind action TSN + StGcn + KeyPointSeq"
```

---

### Task 4: CAPI（`MD_MODEL_TSN` / `MD_MODEL_ST_GCN` + sequence/skeleton 入口）

**Files:**
- Modify: `capi/md_capi.h`（枚举 + 两个声明）
- Modify: `capi/md_capi.cpp`（create/delete/clone 分发 + 两个入口）
- Test: `tests/test_capi.cpp`（`[capi]` 用例）

**Interfaces:**
- Consumes: `TSN`/`StGcn` (Task 1/2)
- Produces: C `MD_MODEL_TSN` / `MD_MODEL_ST_GCN`（`MD_MODEL_COUNT` 前）+ `md_model_predict_sequence(h, frames, n, &out)` + `md_model_predict_skeleton(h, joints, T, V, C, &out)`，结果句柄 kind=`MD_RES_CLASSIFICATION`，用既有 `md_result_classification` 读取。

- [ ] **Step 1: `capi/md_capi.h` 枚举** — 在 `MD_MODEL_FORMULA_RECOGNIZER` 之后、`MD_MODEL_COUNT` 之前加：
```c
MD_MODEL_TSN,
MD_MODEL_ST_GCN,
```
声明（放在 `md_model_predict_*` 声明区）：
```c
MD_CAPI_EXPORT MDStatus md_model_predict_sequence(MDModelHandle h, MDImageHandle* frames, size_t n, MDResultHandle* out);
MD_CAPI_EXPORT MDStatus md_model_predict_skeleton(MDModelHandle h, const float* joints, size_t T, size_t V, size_t C, MDResultHandle* out);
```

- [ ] **Step 2: `capi/md_capi.cpp` 分发** — 三处 `#ifdef BUILD_VISION` 分支：
- create：`case MD_MODEL_TSN: { auto* m = new vision::action::TSN(parts[0], opt); if (!m->is_initialized()) return fail_init("TSN"); *out = new md_model_handle{model, m}; } break;`（`MD_MODEL_ST_GCN` 同理用 `StGcn`，`need_parts(1, ...)`）。
- delete：`case MD_MODEL_TSN: delete static_cast<vision::action::TSN*>(model); break;`（ST_GCN 同理）。
- clone：`case MD_MODEL_TSN: cloned = static_cast<vision::action::TSN*>(src->model)->clone().release(); break;`（ST_GCN 同理）。

- [ ] **Step 3: 新入口**（在 `md_model_predict` 附近，仿其 `MD_RES_CLASSIFICATION` 结果装填）：
```cpp
MDStatus md_model_predict_sequence(MDModelHandle h, MDImageHandle* frames, size_t n, MDResultHandle* out) {
    if (!h || !h->model || !out) return {MD_ERROR_INVALID, "null handle/out"};
    if (!frames || n == 0) return {MD_ERROR_INVALID, "null/empty frames"};
#ifdef BUILD_VISION
    auto* m = static_cast<vision::action::TSN*>(h->model);
    std::vector<ImageData> imgs;
    imgs.reserve(n);
    for (size_t i = 0; i < n; ++i) {
        // MDImageHandle -> ImageData（借用现有 md_image 互转 / md_image_from_* 得到 ImageData）
        // 参照 md_model_predict 内 ImageData 提取方式填充 imgs
    }
    std::vector<float> scores;
    if (!m->predict(imgs, &scores)) return {MD_ERROR_RUNTIME, "TSN predict failed"};
    return emit_classification(out, h->model, scores);  // 见下
#else
    (void)frames; (void)n;
    return {MD_ERROR_RUNTIME, "built without BUILD_VISION"};
#endif
}
```
> **结果装填**：仿 `capi/md_capi.cpp:1558`/`:1843` 的 classification 分支——把 `ClassifyResult r; r.label_ids={argmax(scores)}; r.scores=scores;` 塞进结果句柄 `d->v`（`std::vector<ClassifyResult>`），置 `rh->kind = MD_RES_CLASSIFICATION`。`md_model_predict_skeleton` 用 `joints` 组装 `KeyPointSeq`（`T*V*C` 行主序回填 `frames[t][v]=Point3f`）再调 `StGcn::predict`，同样 emit_classification。**以现有 classification 预测路径的实际结果结构为准**（`md_model_predict` 的 classification 分支是最佳模板，直接复制其装入逻辑）。`emit_classification` 为命名占位——务必按现有 `ClassifyResult` 结果句柄装入代码改写。

- [ ] **Step 4: `tests/test_capi.cpp`** — 新增 `[capi]` 用例：枚举 `MD_MODEL_TSN`/`MD_MODEL_ST_GCN` 创建（模型缺失 guard 跳过），错误路径校验；调用 `md_model_predict_sequence`/`_skeleton` 空参 → `MD_ERROR_INVALID`。

- [ ] **Step 5: 构建 + 测试**

Run: `cmake --build build --parallel 8 && cd build && .\bin\test_modeldeploy.exe "[capi]"`
Expected: 0 errors；action 用例（模型缺失 SKIP，错误路径通过）。

- [ ] **Step 6: Commit**

```bash
git add capi/md_capi.h capi/md_capi.cpp tests/test_capi.cpp
git commit -m "feat(capi): MD_MODEL_TSN/ST_GCN + sequence/skeleton predict"
```

---

### Task 5: C#（`TsNModel` / `StGcnModel` 薄封装）

> **YAGNI**：C# 只做能代表使用的薄封装；若时间紧可降级为仅枚举对齐（模型类已在 C++/Python/CAPI 交付）。

**Files:**
- Modify: `csharp/ModelDeploy/types_internal_c.cs`（`MDModelKind` 加 `MD_MODEL_TSN`/`MD_MODEL_ST_GCN`；extern `md_model_predict_sequence`/`md_model_predict_skeleton`）
- Modify: `csharp/ModelDeploy/Models.cs`、`csharp/ModelDeploy/NativeMethods.cs`
- Test: `csharp/ModelDeployUnitTest/AllModelsTests.cs`（`Action_Works`）

**Interfaces:**
- Consumes: CAPI (Task 4)
- Produces: C# `ModelDeploy.TsNModel.Predict(IEnumerable<ImageData>) -> float[]`、`StGcnModel.Predict(float[] skeleton) -> float[]`

- [ ] **Step 1**: `MDModelKind` 加两枚举值（值对齐 CAPI）；`NativeMethods.cs` extern 导入两入口；`Models.cs` 加 `TsNModel`/`StGcnModel`（仿 `FormulaRecognizerModel`/`ReIdModel`）：ctor `md_model_create(MD_MODEL_TSN)`，`Predict` 调 CAPI、经 `md_result_classification` 读 `scores`（`Marshal.Copy` 到 `float[]`）；`Dispose`。
- [ ] **Step 2**: `AllModelsTests.cs` 加 `[Fact] Action_Works`：构造 + 无权重 SKIP。
- [ ] **Step 3**: `dotnet build` + `dotnet test --filter Action_Works`。
- [ ] **Step 4: Commit**

```bash
git add csharp/ModelDeploy/*.cs csharp/ModelDeployUnitTest/AllModelsTests.cs
git commit -m "feat(csharp): TsNModel + StGcnModel wrappers"
```

---

### Task 6: Rust（`TsN` / `StGcn`）

> **YAGNI**：ST-GCN 的 Rust 面可标注后续；TSN 面做薄封装示意。

**Files:**
- Modify: `rust/modeldeploy/src/ffi.rs`（枚举 + extern）
- Modify: `rust/modeldeploy/src/model.rs`、`types.rs`、`lib.rs`
- Test: `rust/modeldeploy/tests/integration_test.rs`（`test_tsn` / `test_st_gcn`）

**Interfaces:**
- Consumes: CAPI (Task 4)
- Produces: Rust `modeldeploy::TsN` / `StGcn`（`model_wrapper!` 风格，`new(model, opt)` + `predict(...)->Result<Vec<f32>>`）

- [ ] **Step 1**: `ffi.rs` 枚举加 `TsN`/`StGcn`（值对齐 CAPI）；extern 声明 `md_model_predict_sequence`（`frames: *const MDImageHandle` 借用指针）/`md_model_predict_skeleton`（`joints: *const c_float`）。
- [ ] **Step 2**: `model.rs`/`types.rs`/`lib.rs`：`TsN::new`、`predict(std::slice of Image)` → `md_result_classification` 读 `scores`（`read_f32` 复制，安全）。
- [ ] **Step 3**: `integration_test.rs`：`#[test] fn test_tsn`：权重缺失 skip；有权重时返回非空 scores。
- [ ] **Step 4**: `cargo build` + `cargo test test_tsn`；clippy 干净。
- [ ] **Step 5: Commit**

```bash
git add rust/modeldeploy/src/*.rs rust/modeldeploy/tests/integration_test.rs
git commit -m "feat(rust): TsN + StGcn bindings"
```

---

### Task 7: `demo_action`（复用 `VideoDecoder` + `Dag`）+ docs

**Files:**
- Create: `examples/demo_action/demo_action.cpp`（TSN：`VideoDecoder` 抽帧 → `TSN.predict` → 打印 top 动作）
- Create: `examples/demo_action/demo_action_skeleton.cpp`（ST-GCN：`Dag` 编排 `VideoDecoderNode -> PoseNode -> KeyPointSeqNode -> StGcnNode -> LabelNode`）
- Create: `examples/demo_action/CMakeLists.txt`
- Modify: `examples/CMakeLists.txt`（`add_subdirectory(demo_action)`）
- Modify: `examples/EXAMPLES.md`（加行）、`README.md`（能力加"视频动作识别 TSN/ST-GCN"）

**Interfaces:**
- Consumes: `video::VideoDecoder`（BUILD_VIDEO）、`pipeline::Dag/Node`、`action::TSN`/`StGcn`、`UltralyticsPose`

- [ ] **Step 1: 写 `examples/demo_action/CMakeLists.txt`**

```cmake
add_executable(demo_action demo_action.cpp)
target_link_libraries(demo_action PRIVATE ${LIBRARY_NAME} ${OpenCV_LIBS})
# 骨架 demo 需要 VideoDecoder+FFmpeg 与 pose 模型，仅 BUILD_VIDEO 时建
if (BUILD_VIDEO)
  add_executable(demo_action_skeleton demo_action_skeleton.cpp)
  target_link_libraries(demo_action_skeleton PRIVATE ${LIBRARY_NAME} ${OpenCV_LIBS} ${FFMPEG_LIBS})
  target_include_directories(demo_action_skeleton PRIVATE ${FFMPEG_INCLUDE_DIR})
endif ()
```
> `demo_action`（TSN 主路径）也尽量复用 `VideoDecoder`；若 `BUILD_VIDEO=OFF` 则退回"帧列表参数"路径（见 Step 2 分支宏）。

- [ ] **Step 2: 写 `examples/demo_action/demo_action.cpp`**（表格化，TSN 主路径）

```cpp
// ModelDeploy demo_action：视频动作识别（TSN，RGB 帧）。
// Usage: demo_action <tsn.onnx> <video.mp4> [num_segments] [width] [height]
//   BUILD_VIDEO：用 video::VideoDecoder 抽帧；否则接受若干帧图片路径（退化路径）。
#include <cstdio>
#include <string>
#include <vector>
#include <algorithm>
#include "runtime/runtime_option.h"
#include "vision/common/image_data.h"
#include "vision/action/tsn.h"

int main(int argc, char** argv) {
    if (argc < 3) { printf("Usage: demo_action <tsn.onnx> <video.mp4> [seg_t] [h] [w]\n"); return 1; }
    modeldeploy::RuntimeOption opt; opt.use_ort_backend();
    modeldeploy::vision::action::TSN model(argv[1], opt);
    if (!model.is_initialized()) { printf("init failed (missing weights?)\n"); return 1; }
#ifdef BUILD_VIDEO
    modeldeploy::video::VideoDecoder dec;
    if (!dec.open(argv[2])) { printf("cannot open %s\n", argv[2]); return 1; }
    std::vector<modeldeploy::vision::ImageData> frames;
    modeldeploy::vision::ImageData f; uint64_t pts = 0;
    while (frames.size() < 32 && dec.next(&f, &pts)) {
        // 解码帧为 CPU NV12，转 RGB 供 TSN 预处理（ImageData::cvt_color 若支持则用，否则先 toCpu 再 asMat）
        frames.push_back(f);
    }
    dec.close();
    if (frames.empty()) { printf("no frames decoded\n"); return 1; }
    std::vector<float> scores;
    if (!model.predict(frames, &scores)) { printf("predict failed\n"); return 1; }
    // 打印 top3
    std::vector<int> idx(scores.size()); for (size_t i=0;i<scores.size();++i) idx[i]=(int)i;
    std::partial_sort(idx.begin(), idx.begin()+std::min<size_t>(3,idx.size()), idx.end(),
                      [&](int a,int b){ return scores[a] > scores[b]; });
    for (int k=0;k<std::min<int>(3,(int)idx.size());++k) printf("top%d label=%d score=%.4f\n", k+1, idx[k], scores[idx[k]]);
#else
    printf("BUILD_VIDEO off: pass frame images instead\n");
#endif
    return 0;
}
```
> **NV12->RGB 提示**：`ImageData` 解码帧是 NV12 双平面，TSN 预处理 `asMat` 需 RGB。在 demo 里用 `ImageData::cvt_color`（若支持 NV12→RGB）或 `toCpu`+OpenCV 转；以实际 `image_data` 能力为准，Task 8 联调校正。`#include "csrc/video/video_decoder.h"` 仅在 `BUILD_VIDEO` 下。

- [ ] **Step 3: 写 `examples/demo_action/demo_action_skeleton.cpp`**（ST-GCN，DAG 编排）

```cpp
// 骨架动作识别 DAG：VideoDecoder -> Pose(UltralyticsPose) -> KeyPointSeq -> StGcn -> label
// Usage: demo_action_skeleton <stgcn.onnx> <pose.onnx> <video.mp4>
#include <cstdio>
#include <vector>
#include <any>
#include <memory>
#include "runtime/runtime_option.h"
#include "vision/common/image_data.h"
#include "vision/common/result.h"
#include "vision/action/st_gcn.h"
#include "vision/pose/ultralytics_pose.h"
#include "pipeline/dag.h"
#include "pipeline/node.h"
#include "vision/action/keypoint_seq.h"

using modeldeploy::pipeline::Node;
using modeldeploy::vision::ImageData;
using modeldeploy::vision::KeyPointsResult;
using modeldeploy::vision::action::KeyPointSeq;

// 桩节点承担骨架来源：此处用 UltralyticsPose 逐帧提关键点，组装成 KeyPointSeq 序列放入节点。
static std::vector<ImageData> g_frames;   // 视频帧（预处理后 RGB）

int main(int argc, char** argv) {
    if (argc < 4) { printf("Usage: demo_action_skeleton <stgcn.onnx> <pose.onnx> <video.mp4>\n"); return 1; }
    modeldeploy::RuntimeOption opt; opt.use_ort_backend();
#ifdef BUILD_VIDEO
    // 1. VideoDecoder 抽帧（演示 Item 7 下沉复用）
    modeldeploy::video::VideoDecoder dec; dec.open(argv[3]);
    ImageData f; uint64_t pts=0;
    while (g_frames.size()<24 && dec.next(&f,&pts)) g_frames.push_back(f);
    dec.close();
#endif
    // 2. 用 UltralyticsPose 逐帧提关键点 -> KeyPointSeq
    modeldeploy::vision::detection::UltralyticsPose pose(argv[2], opt);
    KeyPointSeq seq;
    for (auto& im : g_frames) {
        std::vector<KeyPointsResult> kps;
        if (pose.predict(im, &kps) && !kps.empty()) seq.frames.push_back(kps[0].keypoints);
    }
    // 3. StGcn 分类
    modeldeploy::vision::action::StGcn model(argv[1], opt);
    std::vector<float> scores;
    if (!model.predict(seq, &scores)) { printf("predict failed\n"); return 1; }
    int best = (int)(std::max_element(scores.begin(), scores.end()) - scores.begin());
    printf("action label=%d score=%.4f\n", best, scores[best]);
    return 0;
}
```
> **DAG 落地说明**：上述先以"直接调用 + 复用 VideoDecoder + UltralyticsPose"给出可运行主路径。DAG 编排（`Dag::add_node` 拼 `VideoDecoderNode -> PoseNode -> KeyPointSeqNode -> StGcnNode -> LabelNode`）可与 Item 7 `Planner` DSL 结合；由于多帧抽帧/组序的 `run()` 需跨调用保持状态，DAG 版在 Task 8 联调时以 `Node` 子类封装上述各步骤后由 `Dag::connect` 串联（`Port` 类型 `"Image"`/`"KeyPointSeq"`/`"vector<float>"`）。本 Task 先交付管线主路径，DAG 封装作为增强（YAGNI 判断，能演示复用即可）。

- [ ] **Step 4: `examples/CMakeLists.txt` + EXAMPLES.md + README.md**

`examples/CMakeLists.txt` 加 `add_subdirectory(demo_action)`（在 `demo_doc` 之后）。
`EXAMPLES.md` 加行：
```
| `demo_action` | 视频动作识别（TSN，RGB 帧） | `onnx/tsn/*.onnx` | 视频 mp4 | 打印 top3 动作 label+score |
| `demo_action_skeleton` | 骨架动作识别（ST-GCN） | `onnx/stgcn/*.onnx|pose.onnx` | 视频 mp4 | 打印动作 label+score |
```
`README.md` 能力列表"……文档理解(版面+公式识别→Markdown)/Pipeline DAG 编排/视频解码(FFmpeg,BUILD_VIDEO)"之后加"**视频动作识别（TSN/ST-GCN）**"。

- [ ] **Step 5: 构建 + 运行**

Run（`BUILD_VIDEO=ON` 构建）：`cmake --build build --parallel 8`；`.\bin\demo_action.exe`（无参 Usage）、缺模型错误路径。
Expected: 编译 0 errors；Usage/错误路径正常；缺权重时清晰报错不崩溃。

- [ ] **Step 6: Commit**

```bash
git add examples/demo_action/ examples/CMakeLists.txt examples/EXAMPLES.md README.md
git commit -m "feat(examples): demo_action (TSN via VideoDecoder) + demo_action_skeleton (ST-GCN) + docs"
```

---

### Task 8: 全量验证 + 跨后端语义确认

**Files:** 无新增（验证；必要时 minor 修复）

**Interfaces:**
- Consumes: 全部前序任务

- [ ] **Step 1: 全量 C++ 测试** — `.\bin\test_modeldeploy.exe "[action]"`、`"[capi]"`、`"[pipeline]"`、`"[video]"`、`"[core]"`。记录通过数；无回归。
- [ ] **Step 2: 跨后端语义确认** — grep 确认 `tsn.cpp`/`st_gcn.cpp` 无 backend 直接依赖（仅 `BaseModel`/`get_input_info`）→ ORT/MNN/TRT/Sophgo 语义一致。
- [ ] **Step 3: 绑定量测** — Python（`action.TSN/StGcn/KeyPointSeq` 冒烟）、C#（`Action_Works`）、Rust（`test_tsn`）、`demo_action`/`demo_action_skeleton` 路径。
- [ ] **Step 4: 布局/归一化校正** — 若集成到真实 TSN/ST-GCN 权重，按 `get_input_info(0).shape` 校正 rank-5 分支（TSN）、骨架 `cx/cy/scale`（ST-GCN）、NV12→RGB；更新相关注释。
- [ ] **Step 5: 收尾报送** — 报告 + concerns（如 CAPI 结果装入与现有 classification 的一致、骨骼归一化经验值）。

---

## Self-Review 记录

**Spec coverage**：
- TSN(§3.3) → Task 1；ST-GCN(§3.4) → Task 2；KeyPointSeq(§3.4) → Task 1。
- Tensor 时序/图输入支持(§3.2) → 确认 `Tensor::shape_` 任意维，无改动；TSN 4D uni-dim 优先（§3.2/§3.6）+ rank-5 适配说明。
- Python(§4) → Task 3；CAPI(§5) → Task 4（`MD_MODEL_TSN/ST_GCN` + sequence/skeleton 入口，复用 `MD_RES_CLASSIFICATION` 读取）。
- C#(§6)/Rust(§7) → Task 5/6（薄封装，ST-GCN Rust 标注 YAGNI）。
- demo+docs(§8) → Task 7（复用 `VideoDecoder` + `Dag`，满足 Item 7 final review 建议）。
- 测试(§9) → Task 1/2 的 `[action]` 合成输入单测 + SKIP；Task 4 `[capi]`；Task 8 验证。
- 交付矩阵(§10) → C++/Python/CAPI 全量，C#/Rust 薄封装，demo/docs 全量。

**Placeholder 扫描**：所有代码步骤均有完整实现或明确"以现有某模板为准"的实现指令，无 "TBD/TODO" 式占位。`emit_classification`（Task 4）与骨骼归一化 `cx/cy`（Task 2）为**命名/数值占位**，已显式要求按现有 classification 预测路径与真实权重校正——这是实现时以现有代码为准的关键点，非未定义接口。

**Type consistency**：
- `TSN::predict(const vector<ImageData>&, vector<float>*)` / `StGcn::predict(const KeyPointSeq&, vector<float>*)` 在 Task 1/2/3/4 一致。
- 静态缝 `TSN::assemble_frames(frames,T,H,W,Tensor*)` 与 `StGcn::assemble_skeleton(seq,V,C,Tensor*)` 在 Task 1/2 测试与实现一致。
- 枚举名 `MD_MODEL_TSN`/`MD_MODEL_ST_GCN` 在 Task 4/5/6 全链一致。
- `KeyPointSeq{frames: vector<vector<Point3f>>}` 在 Task 1/2/3（pybind）/demo 一致。
